from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
SCRIPT = (
    ROOT.parent
    / "优化阶段一"
    / "基于地层约束的测井裂缝预测"
    / "run_conventional_log_strata_validation.py"
)
STAGE1_LIBRARY_DIR = ROOT / "training_runs_stage2_imaging_only_labeled" / "expert_library" / "inner_stage1_library"
STAGE2_LIBRARY_DIR = ROOT / "training_runs_stage2_imaging_only_labeled" / "expert_library" / "inner_stage2_library"
LAYER_DIR = ROOT.parents[2] / "原始数据" / "wx数据" / "砂砾岩" / "层位"
INPUT_ROOT = ROOT / "near_well_attribute_samples_logs_stage1_compatible"
OUTPUT_ROOT = ROOT / "training_runs_stage2_imaging_only_labeled" / "conventional_prediction"

REQUESTED_FEATURES_JSON = "[\"AC\",\"CAL\",\"CNL\",\"DEN\",\"GR\",\"RFOC\",\"RILD\",\"RILM\",\"SP\"]"
EXCLUDED_WELLS = {
    "\u8f66103",
    "\u8f6612",
}
SUMMARY_CSV_NAME = "batch_prediction_run_summary.csv"


def discover_sample_csvs(root: Path) -> list[Path]:
    csvs: list[Path] = []
    for path in sorted(root.rglob("*_stage1_compatible.csv")):
        if path.is_file():
            csvs.append(path)
    return csvs


def main() -> int:
    if not SCRIPT.exists():
        raise FileNotFoundError(f"Prediction script not found: {SCRIPT}")
    if not STAGE1_LIBRARY_DIR.exists():
        raise FileNotFoundError(f"Stage1 library dir not found: {STAGE1_LIBRARY_DIR}")
    if not STAGE2_LIBRARY_DIR.exists():
        raise FileNotFoundError(f"Stage2 library dir not found: {STAGE2_LIBRARY_DIR}")
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sample_csvs = discover_sample_csvs(INPUT_ROOT)
    if not sample_csvs:
        raise FileNotFoundError(f"No *_stage1_compatible.csv files found under {INPUT_ROOT}")

    filtered_csvs: list[Path] = []
    skipped_wells: list[str] = []
    for csv_path in sample_csvs:
        well_name = csv_path.stem.replace("_near_well_attribute_0p2ms_stage1_compatible", "").strip()
        if well_name in EXCLUDED_WELLS:
            skipped_wells.append(well_name)
            continue
        filtered_csvs.append(csv_path)
    sample_csvs = filtered_csvs
    if skipped_wells:
        print("Skipping wells confirmed outside the valid T4-T7 interval:")
        print(", ".join(skipped_wells))

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    summary_rows: list[dict[str, str | int]] = []

    for csv_path in sample_csvs:
        well_name = csv_path.stem.replace("_near_well_attribute_0p2ms_stage1_compatible", "").strip()
        result_dir = OUTPUT_ROOT / f"stage2_new_t4t7_{well_name}"
        cmd = [
            PYTHON,
            "-X",
            "utf8",
            str(SCRIPT),
            "--exp-id",
            f"stage2_new_t4t7_{well_name}",
            "--target-sample-csv",
            str(csv_path),
            "--target-well-name",
            well_name,
            "--layer-dir",
            str(LAYER_DIR),
            "--stage1-library-dir",
            str(STAGE1_LIBRARY_DIR),
            "--stage2-library-dir",
            str(STAGE2_LIBRARY_DIR),
            "--output-root",
            str(OUTPUT_ROOT),
            "--docx-path",
            str(OUTPUT_ROOT / "batch_report.docx"),
            "--requested-log-features-json",
            REQUESTED_FEATURES_JSON,
            "--strata-determination-mode",
            "always_auto",
            "--strata-determination-topk",
            "2",
            "--unknown-strata-policy",
            "skip",
            "--save-debug",
            "0",
        ]
        print("启动命令:")
        print(" ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
        summary_rows.append(
            {
                "WellName": well_name,
                "InputCsv": str(csv_path),
                "ResultDir": str(result_dir),
                "ReturnCode": int(rc),
                "RunStatus": "ok" if rc == 0 else "failed",
            }
        )
        if rc != 0:
            print(f"[WARN] prediction failed for well: {well_name}, returncode={rc}. Continue to next well.")

    summary_csv = OUTPUT_ROOT / SUMMARY_CSV_NAME
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["WellName", "InputCsv", "ResultDir", "ReturnCode", "RunStatus"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    failed_count = sum(1 for row in summary_rows if int(row["ReturnCode"]) != 0)
    print(f"Batch finished. total={len(summary_rows)}, failed={failed_count}, summary={summary_csv}")
    return 0 if failed_count < len(summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
