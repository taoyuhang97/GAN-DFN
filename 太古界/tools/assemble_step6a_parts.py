#!/usr/bin/env python3
"""把 Step6A 预测的**按块并行 part** 拼装成整卷（F3）。

配合 `predict_taigu_density_volume.py --block-start/--block-end --output-dir` 使用：
每个 part 目录里是一份"只含自己那批块"的完整产物
（`predicted_fracture_density.sgy` / `window_code.npz` / `trace_mapping.npz` /
`prediction_block_qc.csv` / `prediction_summary.json`）。

由于块是按 IX 顺序切的、块内按 TraceIdx 排序，而 grid 本身按 TraceIdx 升序，
**按 block 顺序拼接各 part 就等于原始整卷顺序**，因此这里做的是顺序拼接。

用法::

    python3 太古界/tools/assemble_step6a_parts.py \
      --parts-root <part 目录的父目录> \
      --output-dir <整卷输出目录> \
      [--delete-parts]

校验：
* 各 part 的 `block_range` 必须连续且合起来覆盖 1..total_blocks；
* 各 part 的采样轴必须一致；
* 拼接后的道数 = 各 part 道数之和，且 = grid 道数（用 summary 里的 total 推断）；
* 输出 `prediction_summary.json` 的 `status` 取各 part 的与；
  并附 `part_assembly` 段记录每个 part 的块范围与道数。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import segyio


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predict_module():
    """加载 Step6A 预测脚本，复用其层位填充逻辑（保证口径与整卷运行一致）。

    2026-09-24（修复）：part 模式下每个 part 都会写一份**整卷** horizon_fill_qc.csv，
    但拼装时不会产出整卷版；这里用预测脚本自己的函数重算，避免两套实现漂移。
    """
    predict_path = (
        Path(__file__).resolve().parent.parent
        / "step6a_density_volume"
        / "predict_taigu_density_volume.py"
    )
    if not predict_path.exists():
        raise SystemExit(f"[assemble] 找不到 Step6A 预测脚本：{predict_path}")
    spec = importlib.util.spec_from_file_location("step6a_predict_module", predict_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[assemble] 无法加载 Step6A 预测脚本：{predict_path}")
    module = importlib.util.module_from_spec(spec)
    # 预测脚本会在 import 时把 common/ 加进 sys.path，这里直接执行模块顶层代码。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def regenerate_full_horizon_qc(output_dir: Path, summary: dict) -> Path:
    """按整卷口径重建 horizon_fill_qc.csv（读 config → 格网 + 层位契约 → 最近邻填充）。"""
    config_path = summary.get("config_path")
    if not config_path:
        raise SystemExit("[assemble] summary 缺 config_path，无法重建 horizon_fill_qc.csv")
    config = read_json(Path(config_path))
    demo_grid_csv = Path(config["demo_grid_csv"]).resolve()
    horizon_contract_csv = Path(config["horizon_contract_csv"]).resolve()
    grid = pd.read_csv(demo_grid_csv, encoding="utf-8-sig")
    horizon = pd.read_csv(horizon_contract_csv, encoding="utf-8-sig")
    for frame in (grid, horizon):
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grid = grid.merge(
        horizon[["TraceIdx", "TopTimeMs", "MidTimeMs", "BaseTimeMs", "SurfaceValid"]],
        on="TraceIdx",
        how="left",
    )
    grid = grid.sort_values("TraceIdx").reset_index(drop=True)

    predict = load_predict_module()
    max_fill_distance = float(config.get("max_horizon_fill_distance_m", 25.0))
    min_thickness = float(config.get("min_horizon_thickness_ms", 1.0))
    if bool(config.get("horizon_fill_enabled", True)):
        grid = predict.fill_missing_horizons_nearest(grid, max_fill_distance, min_thickness)
    else:
        grid["OriginalSurfaceValid"] = grid["SurfaceValid"].fillna(0).astype(bool)
        grid["HorizonFillUsed"] = False
        grid["HorizonFillValid"] = False
        grid["HorizonFillSourceTraceIdx"] = np.nan
        grid["HorizonFillDistanceM"] = np.nan
        grid["FilledSurfaceValid"] = grid["OriginalSurfaceValid"]
    # 与预测脚本一致：SurfaceValid 跟随 FilledSurfaceValid，再做一次几何顺序复核，
    # 不通过的仍判为无效（预测脚本里这两步都发生在写 horizon_fill_qc.csv 之前，
    # 漏掉会多出约数百道"看起来有效"的填充道）。
    grid["SurfaceValid"] = grid["FilledSurfaceValid"].astype(bool)
    top = grid["TopTimeMs"].to_numpy(dtype=np.float64)
    mid = grid["MidTimeMs"].to_numpy(dtype=np.float64)
    base = grid["BaseTimeMs"].to_numpy(dtype=np.float64)
    horizon_order_valid = (
        np.isfinite(top)
        & np.isfinite(mid)
        & np.isfinite(base)
        & (top < mid)
        & (mid < base)
        & ((mid - top) >= min_thickness)
        & ((base - mid) >= min_thickness)
    )
    grid.loc[~horizon_order_valid, "FilledSurfaceValid"] = False
    grid.loc[~horizon_order_valid, "SurfaceValid"] = False
    qc_columns = [
        "TraceIdx",
        "IX",
        "IY",
        "X",
        "Y",
        "OriginalSurfaceValid",
        "HorizonFillValid",
        "HorizonFillSourceTraceIdx",
        "HorizonFillDistanceM",
        "FilledSurfaceValid",
        "TopTimeMs",
        "MidTimeMs",
        "BaseTimeMs",
    ]
    horizon_qc_path = output_dir / "horizon_fill_qc.csv"
    grid[qc_columns].to_csv(horizon_qc_path, index=False, encoding="utf-8-sig")
    return horizon_qc_path


def merge_running_stats(items: list[dict]) -> dict:
    """合并 RunningStats 型字典（count/min/max/mean/std）。"""
    items = [item for item in items if isinstance(item, dict) and float(item.get("count", 0.0)) > 0]
    if not items:
        return {}
    total = sum(float(item["count"]) for item in items)
    weight_mean = sum(float(item["count"]) * float(item.get("mean", 0.0)) for item in items) / total
    variance = (
        sum(
            float(item["count"])
            * (float(item.get("std", 0.0)) ** 2 + float(item.get("mean", 0.0)) ** 2)
            for item in items
        )
        / total
        - weight_mean**2
    )
    merged = {
        "count": int(total),
        "mean": weight_mean,
        "std": float(np.sqrt(max(variance, 0.0))),
    }
    mins = [float(item["min"]) for item in items if item.get("min") is not None]
    maxs = [float(item["max"]) for item in items if item.get("max") is not None]
    if mins:
        merged["min"] = min(mins)
    if maxs:
        merged["max"] = max(maxs)
    return merged


def merge_bin_stats(per_part: list[dict]) -> dict:
    """合并 density_stats_by_relative_position 的 {bin: {count, mean}} 结构。"""
    merged: dict = {}
    for item in per_part:
        for bin_name, metrics in (item or {}).items():
            count = float(metrics.get("count", 0.0))
            mean = float(metrics.get("mean", 0.0))
            slot = merged.setdefault(bin_name, {"count": 0.0, "sum": 0.0})
            slot["count"] += count
            slot["sum"] += count * mean
    return {
        bin_name: {
            "count": int(slot["count"]),
            "mean": (slot["sum"] / slot["count"]) if slot["count"] else 0.0,
        }
        for bin_name, slot in merged.items()
    }


def is_part_dir(path: Path) -> bool:
    """只认 F3 part 产物：summary 里必须带 block_range 且 part_mode=true。

    （整卷运行的目录也含 prediction_summary.json，必须排除，否则拼装会误判块范围。）
    已拼装过的目录带 part_assembly 段，同样排除，保证可重复运行。
    """
    summary_path = path / "prediction_summary.json"
    if not summary_path.exists() or not (path / "predicted_fracture_density.sgy").exists():
        return False
    try:
        summary = read_json(summary_path)
    except (ValueError, OSError):
        return False
    if "part_assembly" in summary:
        return False
    block_range = dict(summary.get("block_range", {}))
    return bool(block_range.get("part_mode", False))


def repair_existing_output_dir(output_dir: Path) -> int:
    """修复已存在的整卷目录：重写 output_paths + 重建 horizon_fill_qc.csv。"""
    summary_path = output_dir / "prediction_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"[assemble] {summary_path} 不存在，无法修复")
    summary = read_json(summary_path)
    outputs = {
        "density_sgy": output_dir / "predicted_fracture_density.sgy",
        "window_code_npz": output_dir / "window_code.npz",
        "trace_mapping_npz": output_dir / "trace_mapping.npz",
        "prediction_block_qc_csv": output_dir / "prediction_block_qc.csv",
        "horizon_fill_qc_csv": output_dir / "horizon_fill_qc.csv",
    }
    missing = [key for key, path in outputs.items() if not path.exists()]
    if "horizon_fill_qc_csv" in missing and len(missing) == 1:
        regenerate_full_horizon_qc(output_dir, summary)
        missing = [key for key, path in outputs.items() if not path.exists()]
    if missing:
        raise SystemExit(f"[assemble] 整卷目录缺少产物，无法修复：{missing}")
    summary["output_paths"] = {key: str(path) for key, path in outputs.items()}
    summary["output_paths"]["prediction_summary_json"] = str(summary_path)
    summary.setdefault("checks", {})["declared_outputs_present"] = True
    summary["output_paths_repair"] = (
        "2026-09-24：part 拼接后 output_paths 曾指向已删除的 part 目录，此处改回整卷路径。"
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[assemble] 已修复 {output_dir} 的 output_paths；status={summary.get('status')}")
    return 0 if summary.get("status") == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble Step6A predict parts (F3).")
    parser.add_argument("--parts-root", type=Path, help="part 目录的父目录（修复模式不需要）")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delete-parts", action="store_true", help="拼装校验通过后删除 part 目录")
    parser.add_argument(
        "--repair-output-dir",
        action="store_true",
        help=(
            "只修复**已存在**的整卷目录（part 已被删除时使用）："
            "重写 prediction_summary.json 的 output_paths 为整卷路径，并重建 horizon_fill_qc.csv。"
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if args.repair_output_dir:
        return repair_existing_output_dir(output_dir)
    if args.parts_root is None:
        raise SystemExit("[assemble] 需要 --parts-root（或改用 --repair-output-dir）")
    parts_root = args.parts_root.resolve()
    part_dirs = sorted(
        p
        for p in parts_root.iterdir()
        if p.is_dir() and p.resolve() != output_dir and is_part_dir(p)
    )
    if not part_dirs:
        raise SystemExit(f"[assemble] 在 {parts_root} 下没找到 part 目录")

    records = []
    for path in part_dirs:
        summary = read_json(path / "prediction_summary.json")
        block_range = dict(summary.get("block_range", {}))
        if not block_range:
            raise SystemExit(f"[assemble] {path.name} 缺 block_range，可能不是 F3 part 产物")
        records.append(
            {
                "dir": path,
                "summary": summary,
                "block_start": int(block_range["block_start"]),
                "block_end": int(block_range["block_end"]),
                "total_blocks": int(block_range["total_blocks"]),
                "trace_count": int(block_range["part_trace_count"]),
                "status": str(summary.get("status")),
            }
        )
    records.sort(key=lambda item: item["block_start"])

    total_blocks = records[0]["total_blocks"]
    cursor_block = 1
    for record in records:
        if record["total_blocks"] != total_blocks:
            raise SystemExit("[assemble] 各 part 的 total_blocks 不一致")
        if record["block_start"] != cursor_block:
            raise SystemExit(
                f"[assemble] 块范围不连续：期望 {cursor_block}，实际 {record['block_start']}"
            )
        cursor_block = record["block_end"] + 1
    if cursor_block - 1 != total_blocks:
        raise SystemExit(f"[assemble] 块范围未覆盖全部：覆盖到 {cursor_block - 1}/{total_blocks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_sgy = output_dir / "predicted_fracture_density.sgy"
    tmp_sgy = output_dir / "predicted_fracture_density.sgy.partial"
    total_traces = int(sum(record["trace_count"] for record in records))

    window_blocks = []
    qc_frames = []
    map_parts = []
    sample_axis = None
    with segyio.open(str(records[0]["dir"] / "predicted_fracture_density.sgy"), "r", ignore_geometry=True) as first:
        spec = segyio.spec()
        spec.sorting = first.sorting
        # segyio 的 first.format 是 fmt 枚举，写 bin 字段需要整数
        format_code = int(first.bin[segyio.BinField.Format])
        spec.format = format_code
        spec.samples = np.asarray(first.samples, dtype=np.float32)
        spec.tracecount = total_traces
        delay_ms = int(round(float(spec.samples[0])))
        interval_us = int(round(float(np.median(np.diff(spec.samples))) * 1000.0))
        sample_axis = np.asarray(spec.samples, dtype=np.float64)

    with segyio.create(str(tmp_sgy), spec) as out_handle:
        out_handle.bin[segyio.BinField.Interval] = interval_us
        out_handle.bin[segyio.BinField.Samples] = int(len(spec.samples))
        out_handle.bin[segyio.BinField.Format] = format_code
        cursor = 0
        for record in records:
            part_dir = record["dir"]
            with segyio.open(
                str(part_dir / "predicted_fracture_density.sgy"), "r", ignore_geometry=True
            ) as part_handle:
                part_samples = np.asarray(part_handle.samples, dtype=np.float64)
                if len(part_samples) != len(sample_axis) or not np.allclose(
                    part_samples, sample_axis, atol=1.0e-6
                ):
                    raise SystemExit(f"[assemble] {part_dir.name} 的采样轴与第一个 part 不一致")
                for index in range(part_handle.tracecount):
                    header = dict(part_handle.header[index])
                    header[segyio.TraceField.TRACE_SEQUENCE_FILE] = cursor + 1
                    header[segyio.TraceField.TRACE_SEQUENCE_LINE] = cursor + 1
                    header[segyio.TraceField.DelayRecordingTime] = delay_ms
                    out_handle.header[cursor] = header
                    out_handle.trace[cursor] = np.asarray(part_handle.trace[index], dtype=np.float32)
                    cursor += 1
            window_blocks.append(np.load(part_dir / "window_code.npz")["window_code"])
            qc_path = part_dir / "prediction_block_qc.csv"
            if qc_path.exists():
                qc_frames.append(pd.read_csv(qc_path, encoding="utf-8-sig"))
            mapping = np.load(part_dir / "trace_mapping.npz")
            map_parts.append(
                {
                    "trace_idx": mapping["trace_idx"],
                    "x": mapping["x"],
                    "y": mapping["y"],
                }
            )
        if cursor != total_traces:
            raise SystemExit(f"[assemble] 拼接道数 {cursor} != 预期 {total_traces}")
    tmp_sgy.replace(out_sgy)

    window_code = np.concatenate(window_blocks, axis=0)
    np.savez_compressed(
        output_dir / "window_code.npz",
        window_code=window_code,
        sample_axis=sample_axis.astype(np.float32),
    )
    np.savez_compressed(
        output_dir / "trace_mapping.npz",
        trace_idx=np.concatenate([item["trace_idx"] for item in map_parts]),
        x=np.concatenate([item["x"] for item in map_parts]),
        y=np.concatenate([item["y"] for item in map_parts]),
        output_trace_index=np.arange(total_traces, dtype=np.int64),
    )
    if qc_frames:
        qc = pd.concat(qc_frames, ignore_index=True).sort_values("BlockIndex")
        # 原脚本里 Upper/CrustPredictionCount 是**累计**计数（layer_total 累加），
        # 分 part 后每个 part 各自从 0 开始累计；这里按 part 边界还原增量后再整体累计。
        increments: dict[str, list[float]] = {}
        for column in ("UpperPredictionCount", "CrustPredictionCount"):
            if column not in qc.columns:
                continue
            series: list[float] = []
            for frame in qc_frames:
                if column not in frame.columns:
                    continue
                values = frame.sort_values("BlockIndex")[column].to_numpy(dtype=float)
                series.extend(np.diff(values, prepend=0.0).tolist())
            increments[column] = series
        for column, series in increments.items():
            if len(series) == len(qc):
                qc[column] = np.cumsum(np.asarray(series, dtype=float))
        qc.to_csv(output_dir / "prediction_block_qc.csv", index=False, encoding="utf-8-sig")

    merged = dict(records[0]["summary"])
    merged["status"] = "pass" if all(record["status"] == "pass" for record in records) else "fail"
    merged["part_assembly"] = {
        "parts_root": str(parts_root),
        "part_count": int(len(records)),
        "total_blocks": int(total_blocks),
        "total_traces": int(total_traces),
        "parts": [
            {
                "dir": record["dir"].name,
                "block_start": record["block_start"],
                "block_end": record["block_end"],
                "trace_count": record["trace_count"],
                "status": record["status"],
            }
            for record in records
        ],
    }
    merged["output_dir"] = str(output_dir)
    # ---- 把各 part 的 summary 聚合成"整卷"口径（下游只读这一份） ----
    summaries = [record["summary"] for record in records]
    audit_parts = [dict(item.get("attribute_sampling_audit", {})) for item in summaries]
    merged_audit = dict(audit_parts[0])
    for key in ("window_sample_count", "anttrack_finite_count", "anttrack_minus_one_count"):
        merged_audit[key] = int(sum(int(part.get(key, 0) or 0) for part in audit_parts))
    full_axis = int(merged_audit.get("full_axis_sample_count", 0) or 0)
    window_count = int(merged_audit.get("window_sample_count", 0) or 0)
    merged_audit["redundancy_removed_ratio"] = (
        float(full_axis) / float(window_count) if window_count else None
    )
    merged["attribute_sampling_audit"] = merged_audit
    layer_counts: dict[str, int] = {}
    for item in summaries:
        for layer, value in dict(item.get("prediction_counts_by_layer", {})).items():
            layer_counts[layer] = layer_counts.get(layer, 0) + int(value or 0)
    if layer_counts:
        merged["prediction_counts_by_layer"] = layer_counts
    merged["ext50_sample_count"] = int(
        sum(int(item.get("ext50_sample_count", 0) or 0) for item in summaries)
    )
    merged["density_stats"] = {
        name: merge_running_stats([dict(item.get("density_stats", {})).get(name, {}) for item in summaries])
        for name in dict(merged.get("density_stats", {})).keys()
    }
    relative: dict[str, dict] = {}
    for layer in dict(merged.get("density_stats_by_relative_position", {})).keys():
        relative[layer] = merge_bin_stats(
            [
                dict(item.get("density_stats_by_relative_position", {})).get(layer, {})
                for item in summaries
            ]
        )
    if relative:
        merged["density_stats_by_relative_position"] = relative
    merged["aggregate_scope"] = "summed_over_parts"
    checks = dict(merged.get("checks", {}))
    checks["parts_cover_all_blocks"] = True
    checks["assembled_trace_count_matches_parts"] = True
    # ------------------------------------------------------------------
    # 2026-09-24（修复）：part 的 summary 里 output_paths 指向各自的 part 目录，
    # 直接沿用会让下游（如 Step6B 的输入契约校验）去读已删除的 part 目录。
    # 这里统一改写成**整卷目录**下的真实路径，并顺带补齐整卷 horizon_fill_qc.csv。
    # ------------------------------------------------------------------
    horizon_qc_path = regenerate_full_horizon_qc(output_dir, merged)
    merged["output_paths"] = {
        "density_sgy": str(out_sgy),
        "window_code_npz": str(output_dir / "window_code.npz"),
        "trace_mapping_npz": str(output_dir / "trace_mapping.npz"),
        "prediction_block_qc_csv": str(output_dir / "prediction_block_qc.csv"),
        "prediction_summary_json": str(output_dir / "prediction_summary.json"),
        "horizon_fill_qc_csv": str(horizon_qc_path),
    }
    missing_outputs = [
        key for key, value in merged["output_paths"].items()
        if key != "prediction_summary_json" and not Path(value).exists()
    ]
    if missing_outputs:
        raise SystemExit(f"[assemble] 整卷输出缺失：{missing_outputs}")
    checks["declared_outputs_present"] = True
    merged["checks"] = checks
    (output_dir / "prediction_summary.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[assemble] 完成：{len(records)} 个 part → {total_traces} 道，"
        f"{len(sample_axis)} 采样；输出 {output_dir}；status={merged['status']}"
    )
    if args.delete_parts:
        for record in records:
            shutil.rmtree(record["dir"], ignore_errors=True)
        print(f"[assemble] 已删除 {len(records)} 个 part 目录")
    return 0 if merged["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
