#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RECOVERY_SOURCE_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/单元DFN构建"
)
DEFAULT_PACKAGE_PARENT = DEFAULT_RECOVERY_SOURCE_ROOT / "单元测井裂缝_新层位重拆分"
DEFAULT_OUTPUT_ROOT = DEFAULT_RECOVERY_SOURCE_ROOT / "unit_dfn_recovery_20261113_v1"
BUILD_SCRIPT = THIS_DIR / "build_single_unit_dfn.py"
UNIT_ID_PATTERN = re.compile(r"^BX\d+_BY\d+$")


@dataclass(frozen=True)
class UnitPackage:
    unit_id: str
    package_root: Path
    source_group: str


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def natural_unit_key(unit_id: str) -> tuple[int, int, str]:
    match = re.match(r"^BX(\d+)_BY(\d+)$", unit_id)
    if not match:
        return (sys.maxsize, sys.maxsize, unit_id)
    return (int(match.group(1)), int(match.group(2)), unit_id)


def read_catalog_unit_ids(package_root: Path) -> list[str]:
    catalog_path = package_root / "unit_catalog.csv"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"unit_catalog.csv not found: {catalog_path}")
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "UnitID" not in reader.fieldnames:
            raise ValueError(f"UnitID column not found: {catalog_path}")
        unit_ids = [str(row.get("UnitID", "")).strip() for row in reader]
    unit_ids = [unit_id for unit_id in unit_ids if unit_id]
    if not unit_ids:
        raise ValueError(f"no UnitID rows found: {catalog_path}")
    return unit_ids


def validate_unit_package(package_root: Path, unit_id: str) -> None:
    if not UNIT_ID_PATTERN.match(unit_id):
        raise ValueError(f"invalid UnitID: {unit_id}")
    unit_dir = package_root / "units" / unit_id
    required = [
        unit_dir / "fracture_seeds.csv",
        unit_dir / "unit_layers.csv",
        unit_dir / "unit_meta.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete package for {unit_id}: {missing}")


def package_is_buildable(package_root: Path, unit_id: str) -> tuple[bool, str]:
    catalog_path = package_root / "unit_catalog.csv"
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next((item for item in csv.DictReader(handle) if str(item.get("UnitID", "")).strip() == unit_id), None)
    if row is None:
        return False, "catalog_row_not_found"
    block_x = str(row.get("BlockX", "")).strip()
    block_y = str(row.get("BlockY", "")).strip()
    layer_count = str(row.get("LayerCount", "")).strip()
    data_mode = str(row.get("DataMode", "")).strip()
    if not block_x or not block_y:
        return False, "missing_block_xy"
    try:
        int(float(block_x))
        int(float(block_y))
        layers = int(float(layer_count or "0"))
    except ValueError:
        return False, "invalid_block_xy_or_layer_count"
    if layers <= 0:
        return False, "no_layers"
    if data_mode == "empty":
        return False, "data_mode_empty"
    return True, ""


def load_root_packages(
    package_root: Path, source_group: str
) -> tuple[list[UnitPackage], list[dict[str, str]]]:
    records: list[UnitPackage] = []
    excluded: list[dict[str, str]] = []
    for unit_id in read_catalog_unit_ids(package_root):
        buildable, reason = package_is_buildable(package_root, unit_id)
        if not buildable:
            excluded.append({"UnitID": unit_id, "SourceGroup": source_group, "Reason": reason})
            continue
        validate_unit_package(package_root, unit_id)
        records.append(UnitPackage(unit_id=unit_id, package_root=package_root, source_group=source_group))
    return records, excluded


def load_nested_packages(
    package_parent: Path, source_group: str
) -> tuple[list[UnitPackage], list[dict[str, str]]]:
    if not package_parent.is_dir():
        raise FileNotFoundError(f"package parent not found: {package_parent}")
    records: list[UnitPackage] = []
    excluded: list[dict[str, str]] = []
    for package_root in sorted(package_parent.iterdir(), key=lambda path: natural_unit_key(path.name)):
        if not package_root.is_dir():
            continue
        catalog_path = package_root / "unit_catalog.csv"
        if not catalog_path.is_file():
            continue
        unit_ids = read_catalog_unit_ids(package_root)
        if unit_ids != [package_root.name]:
            raise ValueError(
                f"per-unit package must contain exactly its own UnitID: "
                f"directory={package_root.name} catalog={unit_ids}"
            )
        buildable, reason = package_is_buildable(package_root, package_root.name)
        if not buildable:
            excluded.append({"UnitID": package_root.name, "SourceGroup": source_group, "Reason": reason})
            continue
        record = UnitPackage(
            unit_id=package_root.name,
            package_root=package_root,
            source_group=source_group,
        )
        validate_unit_package(record.package_root, record.unit_id)
        records.append(record)
    return records, excluded


def discover_packages(base_root: Path, package_parent: Path) -> tuple[list[UnitPackage], list[dict[str, str]]]:
    records: list[UnitPackage] = []
    excluded: list[dict[str, str]] = []
    loaded, skipped = load_root_packages(
        package_parent / "full_t1_t7_resplit_20260501",
        "base_full_t1_t7_20260501",
    )
    records.extend(loaded)
    excluded.extend(skipped)
    loaded, skipped = load_nested_packages(
        package_parent / "supplement_blank_units_20260606_per_unit",
        "supplement_130_20260606",
    )
    records.extend(loaded)
    excluded.extend(skipped)
    loaded, skipped = load_root_packages(
        package_parent / "supplement_blank_units_20260605",
        "supplement_1_20260605",
    )
    records.extend(loaded)
    excluded.extend(skipped)

    by_unit: dict[str, UnitPackage] = {}
    duplicates: list[str] = []
    for record in records:
        if record.unit_id in by_unit:
            duplicates.append(record.unit_id)
            continue
        by_unit[record.unit_id] = record
    if duplicates:
        raise ValueError(f"duplicate UnitIDs across package sources: {sorted(set(duplicates))}")
    return sorted(by_unit.values(), key=lambda item: natural_unit_key(item.unit_id)), excluded


def required_outputs(unit_output_dir: Path) -> list[Path]:
    return [
        unit_output_dir / "unit_dfn_summary.json",
        unit_output_dir / "unit_dfn_patches.csv",
        unit_output_dir / "unit_dfn_patches_raw_time.vtk",
        unit_output_dir / "unit_dfn_patches_display.vtk",
    ]


def outputs_complete(unit_output_dir: Path) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in required_outputs(unit_output_dir))


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "UnitID",
        "SourceGroup",
        "PackageRoot",
        "OutputDir",
        "Status",
        "ElapsedSeconds",
        "PatchCount",
        "InputSeedCount",
        "MergedSeedCount",
        "LogPath",
        "ErrorMessage",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    excluded: list[dict[str, str]],
    args: argparse.Namespace,
    started: float,
) -> None:
    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("Status", ""))
        source = str(row.get("SourceGroup", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
    payload = {
        "updated_at": now_text(),
        "status": "completed" if all(row.get("Status") in {"success", "skipped"} for row in rows) else "running_or_failed",
        "output_root": str(Path(args.output_root).resolve()),
        "unit_root": str((Path(args.output_root) / "units").resolve()),
        "total_units": len(rows),
        "excluded_empty_package_count": len(excluded),
        "excluded_empty_packages": excluded,
        "status_counts": status_counts,
        "source_counts": source_counts,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "unit_filter": str(args.unit_id or ""),
        "limit": int(args.limit or 0),
        "max_workers": int(args.max_workers),
        "resume": bool(args.resume),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_unit_result(unit_output_dir: Path) -> dict[str, Any]:
    summary_path = unit_output_dir / "unit_dfn_summary.json"
    if not summary_path.is_file():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_one(
    record: UnitPackage,
    output_unit_root: Path,
    log_dir: Path,
    resume: bool,
    env: dict[str, str],
) -> dict[str, Any]:
    unit_output_dir = output_unit_root / record.unit_id
    log_path = log_dir / f"{record.unit_id}.log"
    base_row: dict[str, Any] = {
        "UnitID": record.unit_id,
        "SourceGroup": record.source_group,
        "PackageRoot": str(record.package_root),
        "OutputDir": str(unit_output_dir),
        "LogPath": str(log_path),
    }
    if resume and outputs_complete(unit_output_dir):
        summary = read_unit_result(unit_output_dir)
        return {
            **base_row,
            "Status": "skipped",
            "PatchCount": summary.get("PatchCount", ""),
            "InputSeedCount": summary.get("InputSeedCount", ""),
            "MergedSeedCount": summary.get("MergedSeedCount", ""),
        }

    command = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--unit-id",
        record.unit_id,
        "--package-run-dir",
        str(record.package_root.resolve()),
        "--output-root",
        str(output_unit_root.resolve()),
        "--overwrite",
    ]
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"command={command}\n")
        log_handle.write(f"started_at={now_text()}\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(BUILD_SCRIPT.parent),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_handle.write(f"finished_at={now_text()}\nreturn_code={completed.returncode}\n")

    elapsed = round(time.monotonic() - started, 3)
    if completed.returncode != 0:
        return {
            **base_row,
            "Status": "failed",
            "ElapsedSeconds": elapsed,
            "ErrorMessage": f"build_single_unit_dfn.py return_code={completed.returncode}",
        }
    if not outputs_complete(unit_output_dir):
        missing = [str(path) for path in required_outputs(unit_output_dir) if not path.is_file() or path.stat().st_size == 0]
        return {
            **base_row,
            "Status": "failed",
            "ElapsedSeconds": elapsed,
            "ErrorMessage": f"required outputs missing or empty: {missing}",
        }

    summary = read_unit_result(unit_output_dir)
    return {
        **base_row,
        "Status": "success",
        "ElapsedSeconds": elapsed,
        "PatchCount": summary.get("PatchCount", ""),
        "InputSeedCount": summary.get("InputSeedCount", ""),
        "MergedSeedCount": summary.get("MergedSeedCount", ""),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild per-unit DFNs from preserved per-unit input packages with resume and audit outputs."
    )
    parser.add_argument("--recovery-source-root", type=Path, default=DEFAULT_RECOVERY_SOURCE_ROOT)
    parser.add_argument("--package-parent", type=Path, default=DEFAULT_PACKAGE_PARENT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--unit-id", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.recovery_source_root = Path(args.recovery_source_root).expanduser().resolve()
    args.package_parent = Path(args.package_parent).expanduser().resolve()
    args.output_root = Path(args.output_root).expanduser().resolve()
    args.max_workers = max(1, int(args.max_workers))

    if not BUILD_SCRIPT.is_file():
        raise FileNotFoundError(f"build script not found: {BUILD_SCRIPT}")

    records, excluded_packages = discover_packages(args.recovery_source_root, args.package_parent)
    if args.unit_id:
        filter_value = str(args.unit_id).strip()
        records = [record for record in records if record.unit_id == filter_value]
        if not records:
            raise ValueError(f"UnitID not found in discovered packages: {filter_value}")
    if args.limit and int(args.limit) > 0:
        records = records[: int(args.limit)]
    if not records:
        raise ValueError("no unit packages selected")

    output_root = args.output_root
    output_unit_root = output_root / "units"
    log_dir = output_root / "logs"
    output_unit_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "recovery_manifest.csv"
    summary_path = output_root / "recovery_summary.json"

    started = time.monotonic()
    rows_by_unit: dict[str, dict[str, Any]] = {
        record.unit_id: {
            "UnitID": record.unit_id,
            "SourceGroup": record.source_group,
            "PackageRoot": str(record.package_root),
            "OutputDir": str(output_unit_root / record.unit_id),
            "LogPath": str(log_dir / f"{record.unit_id}.log"),
            "Status": "pending",
        }
        for record in records
    }
    write_manifest(manifest_path, list(rows_by_unit.values()))
    write_summary(summary_path, list(rows_by_unit.values()), excluded_packages, args, started)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    print(f"[recover] output_root={output_root}", flush=True)
    print(
        f"[recover] units={len(records)} excluded_empty={len(excluded_packages)} "
        f"workers={args.max_workers} resume={args.resume}",
        flush=True,
    )

    completed_count = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {
            executor.submit(
                run_one,
                record,
                output_unit_root,
                log_dir,
                bool(args.resume),
                env,
            ): record
            for record in records
        }
        for future in as_completed(future_map):
            record = future_map[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    **rows_by_unit[record.unit_id],
                    "Status": "failed",
                    "ErrorMessage": repr(exc),
                }
            rows_by_unit[record.unit_id] = row
            completed_count += 1
            ordered_rows = [rows_by_unit[item.unit_id] for item in records]
            write_manifest(manifest_path, ordered_rows)
            write_summary(summary_path, ordered_rows, excluded_packages, args, started)
            print(
                f"[recover] {completed_count}/{len(records)} {record.unit_id} "
                f"status={row.get('Status')} patches={row.get('PatchCount', '')} "
                f"elapsed={row.get('ElapsedSeconds', '')}",
                flush=True,
            )

    final_rows = [rows_by_unit[item.unit_id] for item in records]
    write_manifest(manifest_path, final_rows)
    write_summary(summary_path, final_rows, excluded_packages, args, started)
    failures = [row for row in final_rows if row.get("Status") == "failed"]
    print(
        f"[recover] finished total={len(final_rows)} success={sum(row.get('Status') == 'success' for row in final_rows)} "
        f"skipped={sum(row.get('Status') == 'skipped' for row in final_rows)} failed={len(failures)}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
