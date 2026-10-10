#!/usr/bin/env python3
"""Step6A 预测的按块并行驱动（F3）。

把 Step6A 预测按 x-line 块切成 N 段，用 N 个进程并行计算，每个进程写自己的
part 目录（完整的 SGY + window_code + trace_mapping + block QC + summary），
全部成功后再调用拼装脚本合成整卷。

块与块在输出文件里是连续区间、块内按 TraceIdx 排序，因此**按块顺序拼接 = 整卷**；
已被冒烟验证：2 个 part 拼装后与单进程整卷逐位一致（SGY 采样值/道头、window_code、
block QC（含累计列）、summary 聚合）。

用法::

    python3 太古界/tools/run_step6a_parallel.py \
      --config <predict config> --output-dir <整卷输出> --parts-root <part 父目录> \
      --workers 8 [--keep-parts]

说明：
* `--workers` 默认 8；每个进程内存约 4–6 GB（memmap 属性体 + 块缓冲），8 进程约 40 GB。
* 单个 part 失败即整体失败，part 目录保留供排查；成功时默认删除 part 目录。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
PREDICT = TAIGU / "step6a_density_volume/predict_taigu_density_volume.py"
ASSEMBLE = TAIGU / "tools/assemble_step6a_parts.py"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel Step6A predict driver (F3).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config.resolve())
    grid_path = Path(str(config["demo_grid_csv"])).resolve()
    grid = pd.read_csv(grid_path, usecols=["IX"])
    x_line_count = int(grid["IX"].max()) + 1
    block_x_lines = max(int(config.get("block_x_line_count", 20)), 1)
    total_blocks = (x_line_count + block_x_lines - 1) // block_x_lines
    workers = max(1, min(int(args.workers), total_blocks))
    print(
        f"[parallel] x_lines={x_line_count} block_x_lines={block_x_lines} "
        f"total_blocks={total_blocks} workers={workers}",
        flush=True,
    )

    parts_root = args.parts_root.resolve()
    parts_root.mkdir(parents=True, exist_ok=True)
    log_dir = parts_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 连续切分块区间，保证拼装顺序正确
    bounds = [
        (1 + i * total_blocks // workers, (i + 1) * total_blocks // workers)
        for i in range(workers)
    ]
    bounds = [(start, end) for start, end in bounds if start <= end]

    processes = []
    started = time.time()
    for index, (block_start, block_end) in enumerate(bounds):
        part_dir = parts_root / f"part_{index:02d}_blocks_{block_start:03d}_{block_end:03d}"
        log_path = log_dir / f"{part_dir.name}.log"
        command = [
            sys.executable,
            str(PREDICT),
            "--config",
            str(args.config.resolve()),
            "--output-dir",
            str(part_dir),
            "--block-start",
            str(block_start),
            "--block-end",
            str(block_end),
            "--replace-output",
        ]
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(REPO))
        processes.append((part_dir, log_path, handle, process, (block_start, block_end)))
        print(
            f"[parallel] 启动 {part_dir.name}（块 {block_start}-{block_end}），日志 {log_path}",
            flush=True,
        )

    failed: list[str] = []
    for part_dir, log_path, handle, process, block_range in processes:
        returncode = process.wait()
        handle.close()
        elapsed = time.time() - started
        status = "OK" if returncode == 0 else f"FAILED(rc={returncode})"
        print(
            f"[parallel] {part_dir.name} 块 {block_range[0]}-{block_range[1]} {status}"
            f"（累计 {elapsed/60.0:.1f} min）",
            flush=True,
        )
        if returncode != 0:
            failed.append(str(log_path))

    if failed:
        print("[parallel] 有 part 失败，保留 part 目录与日志：", flush=True)
        for path in failed:
            print("   ", path, flush=True)
        return 1

    print(f"[parallel] 全部 part 完成，用时 {(time.time()-started)/60.0:.1f} min，开始拼装", flush=True)
    command = [
        sys.executable,
        str(ASSEMBLE),
        "--parts-root",
        str(parts_root),
        "--output-dir",
        str(args.output_dir.resolve()),
    ]
    if not args.keep_parts:
        command.append("--delete-parts")
    result = subprocess.run(command, cwd=str(REPO))
    if result.returncode != 0:
        print("[parallel] 拼装失败，part 目录保留", flush=True)
        return result.returncode
    print(f"[parallel] 完成，总用时 {(time.time()-started)/60.0:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
