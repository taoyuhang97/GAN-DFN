# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

from baseline_common import DEFAULT_DOCX_PATH, DEFAULT_OUTPUT_ROOT, DEFAULT_UNIT_DFN_ROOT, append_lines_to_docx


DEFAULT_CHECKPOINT = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容三\G_DFN监督基线\训练结果\quick_cpu_train_v2_20260330\checkpoints\best_model.pt"
)

DEFAULT_SELECTED_UNITS: list[tuple[str, str, str]] = [
    ("BX49_BY5", "anchor_isolated", "孤立实井控制单元，检验跨簇适配性"),
    ("BX60_BY54", "anchor_train_core", "训练主簇核心单元，作为流程基准"),
    ("BX61_BY49", "anchor_test_sparse", "原测试单元，重点检查是否仍出现全零"),
    ("BX60_BY53", "near_train_unseen", "训练邻域未见单元，检查近邻泛化"),
    ("BX61_BY52", "near_train_unseen", "同簇未见单元，检查层内窗口推理稳定性"),
    ("BX62_BY50", "near_train_unseen", "同簇外扩单元，检查一步外推效果"),
    ("BX65_BY40", "real_virtual_unseen", "含真实测井约束的未见单元，作为中部对照"),
    ("BX67_BY36", "real_virtual_unseen", "裂缝量较高且有真实控制，适合做质量对照"),
    ("BX72_BY41", "virtual_only_unseen", "虚拟控制高裂缝单元，检查无真实井条件下表现"),
    ("BX79_BY43", "virtual_only_edge", "东部远端高裂缝单元，检查远端区域外推"),
]

SUMMARY_PATTERNS = {
    "UnitID": re.compile(r'"UnitID"\s*:\s*"([^"]+)"'),
    "BlockX": re.compile(r'"BlockX"\s*:\s*(\d+)'),
    "BlockY": re.compile(r'"BlockY"\s*:\s*(\d+)'),
    "ReliabilityClass": re.compile(r'"ReliabilityClass"\s*:\s*"([^"]+)"'),
    "DataMode": re.compile(r'"DataMode"\s*:\s*"([^"]+)"'),
    "LayerCount": re.compile(r'"LayerCount"\s*:\s*(\d+)'),
    "PatchCount": re.compile(r'"PatchCount"\s*:\s*(\d+)'),
    "RealSeedCount": re.compile(r'"RealSeedCount"\s*:\s*(\d+)'),
    "VirtualSeedCount": re.compile(r'"VirtualSeedCount"\s*:\s*(\d+)'),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a 10-unit small-range validation list for production DFN inference.")
    parser.add_argument("--unit-dfn-root", type=Path, default=DEFAULT_UNIT_DFN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "验证清单")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-name", type=str, default=f"small_validation_units_10_{datetime.now().strftime('%Y%m%d')}")
    return parser


def extract_summary_fields(summary_path: Path) -> dict[str, str]:
    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    row: dict[str, str] = {}
    for key, pattern in SUMMARY_PATTERNS.items():
        match = pattern.search(text)
        row[key] = match.group(1) if match else ""
    return row


def build_output_rows(unit_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for unit_id, group, reason in DEFAULT_SELECTED_UNITS:
        summary_path = unit_root / unit_id / "unit_dfn_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"summary not found for unit {unit_id}: {summary_path}")
        row = extract_summary_fields(summary_path)
        row["UnitID"] = unit_id
        row["SelectionGroup"] = group
        row["SelectionReason"] = reason
        rows.append(row)
    return rows


def write_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "UnitID",
        "BlockX",
        "BlockY",
        "ReliabilityClass",
        "DataMode",
        "LayerCount",
        "PatchCount",
        "RealSeedCount",
        "VirtualSeedCount",
        "SelectionGroup",
        "SelectionReason",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(md_path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# 小范围验证单元清单（10个）",
        "",
        "本批次目的：在不进行全量推理的前提下，先验证放宽解码模式是否能在不同类型单元上稳定产生非零 DFN。",
        "",
        "选择原则：",
        "- 包含 3 个锚点单元：训练核心、原测试单元、孤立单元。",
        "- 包含 3 个训练邻域未见单元：检查近邻泛化。",
        "- 包含 2 个含真实测井约束的中部未见单元：检查 real+virtual 场景。",
        "- 包含 2 个远端 virtual-only 单元：检查外推与高裂缝响应。",
        "",
        "| UnitID | BX | BY | ReliabilityClass | DataMode | LayerCount | PatchCount | RealSeedCount | VirtualSeedCount | Group | Reason |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {UnitID} | {BlockX} | {BlockY} | {ReliabilityClass} | {DataMode} | {LayerCount} | {PatchCount} | "
            "{RealSeedCount} | {VirtualSeedCount} | {SelectionGroup} | {SelectionReason} |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_command(checkpoint: Path, csv_path: Path, run_name: str) -> str:
    return (
        r'py -3.12 .\研究内容三\优化阶段一\G_DFN监督基线\infer_production_units_baseline.py '
        f'--checkpoint "{checkpoint}" '
        f'--unit-id-csv "{csv_path}" '
        '--decode-mode relaxed --relaxed-min-count 1 --center-threshold 0.1 '
        f'--run-name "{run_name}" --device cpu'
    )


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = build_output_rows(Path(args.unit_dfn_root))
    csv_path = output_root / f"{args.run_name}.csv"
    md_path = output_root / f"{args.run_name}.md"
    command_path = output_root / f"{args.run_name}_command.txt"
    command = build_command(Path(args.checkpoint), csv_path, f"baseline_production_relaxed_t01_{args.run_name}")

    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    command_path.write_text(command + "\n", encoding="utf-8")

    append_lines_to_docx(
        Path(args.docx_path),
        "2026-03-31 小范围生产推理验证单元挑选",
        [
            "目标：先用 10 个代表性单元验证放宽解码推理，暂不启动 113 个单元全量推理。",
            f"清单CSV：{csv_path}",
            f"说明文档：{md_path}",
            f"启动命令：{command}",
            "单元组成：BX49_BY5, BX60_BY54, BX61_BY49, BX60_BY53, BX61_BY52, BX62_BY50, BX65_BY40, BX67_BY36, BX72_BY41, BX79_BY43",
            "分组原则：锚点单元 + 训练邻域未见单元 + real_virtual 未见单元 + virtual_only 远端单元。",
        ],
    )

    print(csv_path)
    print(md_path)
    print(command_path)


if __name__ == "__main__":
    main()
