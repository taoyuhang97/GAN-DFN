from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sample_reconstruction_utils import (
    discover_around_data_csvs,
    infer_well_name_from_path,
    resample_around_data_csv,
)


DEFAULT_INPUT_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\测井-地震时窗"
)
DEFAULT_OUTPUT_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\测井-地震时窗_uniform_0p2ms"
)
DEFAULT_SECONDARY_INPUT_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\井斜\测井-地震时窗"
)
DEFAULT_SECONDARY_OUTPUT_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\井斜\测井-地震时窗_uniform_0p2ms"
)
DEFAULT_INTERVAL = 0.2


@dataclass
class ResampleTask:
    input_csv: Path
    output_csv: Path
    well_name: str
    source_group: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_resample_tasks() -> list[ResampleTask]:
    imaging_mappings = [
        ("车660_1_around_data_rebuild.csv", "车660_1_around_data.csv"),
        ("车660_2_around_data.csv", "车660_2_around_data.csv"),
        ("车662_around_data.csv", "车662_around_data.csv"),
        ("车663_around_data.csv", "车663_around_data.csv"),
    ]
    tasks: list[ResampleTask] = []

    for input_name, output_name in imaging_mappings:
        input_csv = DEFAULT_INPUT_DIR / input_name
        output_csv = DEFAULT_OUTPUT_DIR / output_name
        well_name = infer_well_name_from_path(output_csv)
        tasks.append(
            ResampleTask(
                input_csv=input_csv,
                output_csv=output_csv,
                well_name=well_name,
                source_group="成像测井",
            )
        )

    for input_csv in discover_around_data_csvs(DEFAULT_SECONDARY_INPUT_DIR):
        output_csv = DEFAULT_SECONDARY_OUTPUT_DIR / input_csv.name
        well_name = infer_well_name_from_path(output_csv)
        tasks.append(
            ResampleTask(
                input_csv=input_csv,
                output_csv=output_csv,
                well_name=well_name,
                source_group="井斜",
            )
        )

    return tasks


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    interval = float(args.interval)
    overwrite = bool(args.overwrite)

    if interval <= 0:
        raise ValueError(f"interval must be > 0, got: {interval}")
    if not DEFAULT_INPUT_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {DEFAULT_INPUT_DIR}")
    if not DEFAULT_SECONDARY_INPUT_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {DEFAULT_SECONDARY_INPUT_DIR}")

    tasks = build_resample_tasks()
    if not tasks:
        raise FileNotFoundError("No resample tasks were built")

    missing_inputs = [task.input_csv for task in tasks if not task.input_csv.exists()]
    if missing_inputs:
        missing_text = "\n".join(str(path) for path in missing_inputs)
        raise FileNotFoundError(f"Missing required input csv:\n{missing_text}")

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_SECONDARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Primary input dir   : {DEFAULT_INPUT_DIR}")
    print(f"Primary output dir  : {DEFAULT_OUTPUT_DIR}")
    print(f"Secondary input dir : {DEFAULT_SECONDARY_INPUT_DIR}")
    print(f"Secondary output dir: {DEFAULT_SECONDARY_OUTPUT_DIR}")
    print(f"Interval            : {interval} ms")
    print(f"Found tasks         : {len(tasks)}")

    ok_count = 0
    skipped_count = 0
    failed_count = 0

    for task in tasks:
        input_csv = task.input_csv
        output_csv = task.output_csv
        well_name = task.well_name

        if output_csv.exists() and not overwrite:
            skipped_count += 1
            print(f"[SKIP] [{task.source_group}] {well_name}: output exists -> {output_csv}")
            continue

        try:
            stats = resample_around_data_csv(
                input_csv=input_csv,
                output_csv=output_csv,
                interval=interval,
                well_name=well_name,
            )
            ok_count += 1
            print(
                f"[OK] [{task.source_group}] "
                f"{well_name}: "
                f"rows {stats.original_row_count} -> {stats.output_row_count}, "
                f"invalid_time={stats.dropped_invalid_time_count}, "
                f"duplicate_time={stats.duplicate_time_count}, "
                f"time=[{stats.time_min}, {stats.time_max}]"
            )
        except Exception as exc:
            failed_count += 1
            print(f"[FAIL] [{task.source_group}] {well_name}: {exc}")

    print(
        "Done: "
        f"ok={ok_count}, skipped={skipped_count}, failed={failed_count}, total={len(tasks)}"
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
