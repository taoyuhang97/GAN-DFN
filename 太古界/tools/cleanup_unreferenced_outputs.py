#!/usr/bin/env python3
"""清理「太古界」下**旧口径、且当前链条不再引用**的输出产物（2026-09-26）。

## 口径（严格按需求方定义）

* **保留**：
  * 名字含 `quanqu1` 的一切（本轮矿区/全区产物）；
  * 被**当前链条**引用到的一切 —— 引用来源 =
    `太古界/**/configs/**/*quanqu1*.json` 与 `太古界/run_*.sh`；
  * 显式保护名单（见 `PROTECTED_SUBSTRINGS`）：训练模型、Step2/3/4 的 md1 口径、
    Step9 的 azi4 新旧画法对照产物等；
  * 体积 < `--min-mb`（默认 50 MB）的小目录一律不动（收益极小、误删代价不对称）。
* **删除**：其余 `太古界/*/output/<口径目录>`。

砂砾岩目录（`优化阶段二/正式主线/`）不在本脚本扫描范围内，天然不会被碰。

## 用法

    python3 太古界/tools/cleanup_unreferenced_outputs.py                 # dry-run，只列清单
    python3 太古界/tools/cleanup_unreferenced_outputs.py --apply         # 真正删除
    python3 太古界/tools/cleanup_unreferenced_outputs.py --apply --manifest <path.json>

删除前会把完整清单（路径、体积）写进 manifest，便于事后追溯。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "太古界"

# 显式保护：即使当前链条没引用也不删（历史对照 / 训练输入 / 报告素材）
PROTECTED_SUBSTRINGS: tuple[str, ...] = (
    "quanqu1",
    "taigu_step6a_attribute_v3_md1",          # Step6A 训练模型（quanqu1 直接复用）
    "taigu_step2_regular_md1",                # Step9 剖面井轨迹分段
    "taigu_step3_imaging_md1",                # Step9 成像分组
    "taigu_step4_gr_rd_rs_md1",               # Step9 常规测井预测点
    "taigu_step9_multibackground_v2_azi4",    # Step9 azi4 新旧画法对照（问题记录 §0.40）
)


def dir_size_mb(path: Path) -> float:
    out = subprocess.run(["du", "-sm", str(path)], capture_output=True, text=True).stdout
    try:
        return float(out.split()[0])
    except (IndexError, ValueError):
        return 0.0


def reference_blob() -> str:
    sources = list(ROOT.glob("**/configs/**/*quanqu1*.json")) + list(ROOT.glob("run_*.sh"))
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in sources)


def is_referenced(name: str, blob: str) -> bool:
    return any(
        token in blob for token in (f"output/{name}/", f"output/{name}\"", f"output/{name}'")
    )


def collect(min_mb: float) -> tuple[list[dict], list[dict]]:
    blob = reference_blob()
    candidates: list[dict] = []
    kept: list[dict] = []
    for path in sorted(p for p in ROOT.glob("*/output/*") if p.is_dir()):
        name = path.name
        size = dir_size_mb(path)
        record = {"path": str(path.relative_to(REPO)), "size_mb": size, "name": name}
        protected = any(token in name for token in PROTECTED_SUBSTRINGS)
        if protected:
            record["reason"] = "protected"
            kept.append(record)
        elif is_referenced(name, blob):
            record["reason"] = "referenced_by_current_chain"
            kept.append(record)
        elif size < min_mb:
            record["reason"] = f"below_min_mb({min_mb})"
            kept.append(record)
        else:
            record["reason"] = "unreferenced_legacy"
            candidates.append(record)
    return candidates, kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup unreferenced legacy Taigu outputs.")
    parser.add_argument("--apply", action="store_true", help="真正删除（默认只 dry-run）")
    parser.add_argument("--min-mb", type=float, default=50.0, help="小于该体积的目录不动")
    parser.add_argument("--manifest", type=Path, default=None, help="删除清单 JSON 输出路径")
    args = parser.parse_args()

    candidates, kept = collect(args.min_mb)
    total = sum(item["size_mb"] for item in candidates)
    print(f"[cleanup] 保留 {len(kept)} 个目录；删除候选 {len(candidates)} 个，合计 {total / 1024:.2f} GB")
    for item in sorted(candidates, key=lambda entry: -entry["size_mb"]):
        print(f"  {item['size_mb'] / 1024:7.2f} GB  {item['path']}")

    manifest_path = args.manifest or (
        ROOT / "output_quanqu1_logs" / "cleanup_unreferenced_outputs_20260926.json"
    )
    manifest = {
        "generated_at": time.strftime("%F %T"),
        "mode": "apply" if args.apply else "dry_run",
        "min_mb": args.min_mb,
        "total_gb": total / 1024.0,
        "candidates": candidates,
        "kept": kept,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[cleanup] 清单已写：{manifest_path}")

    if not args.apply:
        print("[cleanup] dry-run，未删除任何文件")
        return 0

    freed = 0.0
    for item in candidates:
        target = REPO / item["path"]
        parts = item["path"].split("/")
        # 防御性校验：必须是 太古界/<step>/output/<name> 形状，且不在保护名单里
        if len(parts) != 4 or parts[0] != "太古界" or parts[2] != "output":
            raise SystemExit(f"[cleanup] 拒绝删除（路径形状不符）：{target}")
        if not target.exists() or not target.is_dir():
            raise SystemExit(f"[cleanup] 拒绝删除（不存在或非目录）：{target}")
        if any(token in target.name for token in PROTECTED_SUBSTRINGS):
            raise SystemExit(f"[cleanup] 拒绝删除（命中保护名单）：{target}")
        shutil.rmtree(target)
        freed += item["size_mb"]
    print(f"[cleanup] 已删除 {len(candidates)} 个目录，释放 {freed / 1024:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
