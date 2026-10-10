#!/usr/bin/env python3
"""逐位比较两次 Step6A 预测产物（验证 F1/F2/F3 改动数值中性）。

比较项：
1. SEG-Y：道数、采样轴、每一道的采样值逐位、关键道头字段（SourceX/Y、采样数/间隔/延迟）；
2. window_code.npz：window_code 整块逐位；
3. prediction_block_qc.csv：除耗时类字段外的所有列；
4. prediction_summary.json：status / checks / 关键计数（除耗时）。

用法::

    python3 太古界/tools/compare_step6a_runs.py --reference <dir> --candidate <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import segyio

# 不参与比较的字段（耗时/文件路径等运行相关项）
IGNORE_QC_COLUMNS = {"ElapsedSeconds"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_sgy(reference: Path, candidate: Path) -> list[str]:
    problems: list[str] = []
    with segyio.open(str(reference), "r", ignore_geometry=True) as ref, segyio.open(
        str(candidate), "r", ignore_geometry=True
    ) as cand:
        if ref.tracecount != cand.tracecount:
            problems.append(f"SGY 道数不同: {ref.tracecount} vs {cand.tracecount}")
            return problems
        if len(ref.samples) != len(cand.samples) or not np.array_equal(
            np.asarray(ref.samples), np.asarray(cand.samples)
        ):
            problems.append("SGY 采样轴不同")
            return problems
        mismatch = 0
        header_mismatch = 0
        for index in range(ref.tracecount):
            ref_trace = np.asarray(ref.trace[index], dtype=np.float32)
            cand_trace = np.asarray(cand.trace[index], dtype=np.float32)
            if not np.array_equal(ref_trace, cand_trace, equal_nan=True):
                mismatch += 1
            for field in (
                segyio.TraceField.SourceX,
                segyio.TraceField.SourceY,
                segyio.TraceField.TRACE_SAMPLE_COUNT,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL,
                segyio.TraceField.DelayRecordingTime,
            ):
                if int(ref.header[index][field]) != int(cand.header[index][field]):
                    header_mismatch += 1
                    break
        if mismatch:
            problems.append(f"SGY 有 {mismatch}/{ref.tracecount} 道采样值不一致")
        if header_mismatch:
            problems.append(f"SGY 有 {header_mismatch}/{ref.tracecount} 道道头字段不一致")
        if not problems:
            print(f"  [pass] SGY 逐位一致（{ref.tracecount} 道 x {len(ref.samples)} 采样，含道头）")
    return problems


def compare_window_code(reference: Path, candidate: Path) -> list[str]:
    ref = np.load(reference / "window_code.npz")
    cand = np.load(candidate / "window_code.npz")
    problems: list[str] = []
    if not np.array_equal(ref["window_code"], cand["window_code"]):
        diff = int((ref["window_code"] != cand["window_code"]).sum())
        problems.append(f"window_code 有 {diff} 个元素不一致")
    else:
        print(f"  [pass] window_code 逐位一致（shape={ref['window_code'].shape}）")
    if not np.array_equal(ref["sample_axis"], cand["sample_axis"]):
        problems.append("window_code.npz 的 sample_axis 不一致")
    return problems


def compare_block_qc(reference: Path, candidate: Path) -> list[str]:
    ref_path = reference / "prediction_block_qc.csv"
    cand_path = candidate / "prediction_block_qc.csv"
    if not ref_path.exists() or not cand_path.exists():
        return ["缺少 prediction_block_qc.csv"]
    ref = pd.read_csv(ref_path, encoding="utf-8-sig")
    cand = pd.read_csv(cand_path, encoding="utf-8-sig")
    problems: list[str] = []
    if len(ref) != len(cand):
        problems.append(f"block QC 行数不同: {len(ref)} vs {len(cand)}")
    columns = [c for c in ref.columns if c in cand.columns and c not in IGNORE_QC_COLUMNS]
    for column in columns:
        if ref[column].dtype.kind in "if" and cand[column].dtype.kind in "if":
            if not np.array_equal(
                ref[column].to_numpy(dtype=float),
                cand[column].to_numpy(dtype=float),
                equal_nan=True,
            ):
                problems.append(f"block QC 列不一致: {column}")
        elif not ref[column].astype(str).equals(cand[column].astype(str)):
            problems.append(f"block QC 列不一致: {column}")
    if not problems:
        print(f"  [pass] block QC 一致（{len(ref)} 行 x {len(columns)} 列，已排除耗时列）")
    return problems


def compare_summary(reference: Path, candidate: Path) -> list[str]:
    ref = read_json(reference / "prediction_summary.json")
    cand = read_json(candidate / "prediction_summary.json")
    problems: list[str] = []
    if ref.get("status") != cand.get("status"):
        problems.append(f"summary status 不同: {ref.get('status')} vs {cand.get('status')}")
    ref_checks = dict(ref.get("checks", {}))
    cand_checks = dict(cand.get("checks", {}))
    for key, value in ref_checks.items():
        if key in cand_checks and bool(cand_checks[key]) != bool(value):
            problems.append(f"summary checks.{key} 不同: {value} vs {cand_checks[key]}")
    ref_audit = dict(ref.get("attribute_sampling_audit", {}))
    cand_audit = dict(cand.get("attribute_sampling_audit", {}))
    # F2 后 window_sample_count 语义变为"层窗内点数"，不参与等价比较
    for key in ("window_sample_count", "full_axis_sample_count", "redundancy_removed_ratio"):
        ref_audit.pop(key, None)
        cand_audit.pop(key, None)
    for key, value in ref_audit.items():
        if key not in cand_audit:
            continue
        # F2 后这两个计数改为"层窗内采样点"口径，绝对值必然变小，不作为差异
        if key in {"anttrack_finite_count", "anttrack_minus_one_count"}:
            print(f"  [info] {key} 口径变化（层窗内）: {value} -> {cand_audit[key]}")
            continue
        other = cand_audit[key]
        if isinstance(value, (int, float)) and isinstance(other, (int, float)):
            if float(value) != float(other) and not (
                np.isnan(float(value)) and np.isnan(float(other))
            ):
                problems.append(f"summary.{key} 不同: {value} vs {other}")
        elif value != other:
            problems.append(f"summary.{key} 不同")
    if not problems:
        print(f"  [pass] summary 一致（status={ref.get('status')}，checks/审计口径相同）")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two Step6A predict runs bitwise.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference.resolve()
    candidate = args.candidate.resolve()
    print(f"参考: {reference}")
    print(f"对比: {candidate}")
    problems: list[str] = []
    problems += compare_sgy(
        reference / "predicted_fracture_density.sgy",
        candidate / "predicted_fracture_density.sgy",
    )
    problems += compare_window_code(reference, candidate)
    problems += compare_block_qc(reference, candidate)
    problems += compare_summary(reference, candidate)
    if problems:
        print("\n[FAIL] 发现差异：")
        for item in problems:
            print("  -", item)
        return 1
    print("\n[PASS] 两次运行逐位一致（数值中性）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
