#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按"C 方案（交付浏览版）"打包砂砾岩矿区 v1 + 太古界 quanqu1 的成果，供对外发送。

C 方案口径（已与需求方确认）：
  * 只保留 Step7D / Step8 的几何 VTK、Step9 与 Step9b 的全部图件与汇总表、
    井轨迹三件、Z 镜像 ParaView 展示包；
  * 去掉 Step1–5 的样本表与 Step6 的深度体 `.sgy`；
  * 去掉 Step7/8 的 `*_patches.csv` 等大表明细（保留汇总/QC 小表）。

**路径口径（关键）**：包内所有文件都保留"相对项目根目录"的路径，
例如 `优化阶段二/正式主线/output/...`、`太古界/step7d_.../output/...`。
对方把压缩包解压到项目根目录即可原位还原；换机器后绝对路径变了，
但项目内相对路径不变。

用法::

    python3 scripts/package_handover_delivery.py                 # 默认输出到 <repo>/../delivery
    python3 scripts/package_handover_delivery.py --out-dir /data/share/xxx
    python3 scripts/package_handover_delivery.py --list-only     # 只列举不打包
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO.parent / "delivery"
PACKAGE_TAG = time.strftime("%Y%m%d")

MB = 1024 * 1024

# 目录级跳过：中间缓存与日志
SKIP_DIR_NAMES = {"__pycache__", "logs", ".mplconfig", "section_samples"}
# 后缀级无条件跳过：缓存/日志
ALWAYS_SKIP_SUFFIXES = {".npz", ".log", ".pyc", ".tmp", ".part", ".sgy", ".segy"}
# 表格类后缀：超过阈值则视为"明细表"跳过
TABLE_SUFFIXES = {".csv", ".tsv", ".txt"}


@dataclass
class Spec:
    """一条打包来源：项目内相对目录 + 所属步骤说明。"""

    src_rel: str
    step: str
    note: str = ""
    max_table_mb: float = 1.0  # 表格类文件的保留上限（MB）


@dataclass
class Region:
    name: str
    version: str
    specs: list[Spec] = field(default_factory=list)


REGIONS: list[Region] = [
    Region(
        name="砂砾岩·矿区",
        version="formal_mine_multiscale_flow_v1",
        specs=[
            Spec(
                "优化阶段二/正式主线/output/formal_mine_multiscale_flow_v1/step7d_fused",
                "Step7D 多尺度融合 DFN",
                "融合后的完整 DFN（预测片 + 原始断层三角面）",
            ),
            Spec(
                "优化阶段二/正式主线/output/formal_mine_multiscale_flow_v1/step8_well_correction",
                "Step8 井控校正 + 井轨迹展示件",
                "井控校正后最终 DFN、单尺度 DFN、井轨迹三件、井位平面图",
            ),
            Spec(
                "优化阶段二/正式主线/output/step9_section_presentation/mine_v1",
                "Step9 剖面展示（对外成图，全量/PPT 两档）",
                "20 图 + 红蓝备份 8 图 + section_presentation_summary.json",
            ),
            Spec(
                "优化阶段二/正式主线/output/formal_mine_multiscale_flow_v1/step9_sections",
                "Step9 剖面原始输出",
                "多背景剖面 + 单尺度诊断剖面（不含 section_samples 缓存）",
            ),
            Spec(
                "优化阶段二/正式主线/output/step9b_dfn_density_metrics/formal_mine_multiscale_flow_v1",
                "Step9b 密度量化",
                "沙三/沙四/整体 P33 热图、成像标定图、井位图、300 m 网格表",
                max_table_mb=50.0,
            ),
            Spec(
                "优化阶段二/正式主线/output/dfn_display_zflip/mine_v1",
                "附：ParaView Z 镜像展示包",
                "负 Z 展示口径，保留全部单元属性；含井名标注 CSV 与说明",
            ),
        ],
    ),
    Region(
        name="太古界·矿区/全区",
        version="taigu_step8_attribute_multiscale_v1_quanqu1",
        specs=[
            Spec(
                "太古界/step7d_multiscale_fused_dfn/output/taigu_step7d_fused_v1_quanqu1",
                "Step7D 多尺度融合 DFN",
                "融合后的完整 DFN（预测片 + 原始断层三角面）",
            ),
            Spec(
                "太古界/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1",
                "Step8 井控校正 + 井轨迹展示件",
                "井控校正后最终 DFN、井轨迹三件、井顶点、QC/审计小表",
            ),
            Spec(
                "太古界/step9_sections/output/taigu_step9_multibackground_v2_quanqu1",
                "Step9 剖面展示",
                "overview 10 图 + local_200m 10 图 + section_summary.json",
            ),
            Spec(
                "太古界/step9b_dfn_density_metrics/output/taigu_quanqu1_v1",
                "Step9b 分地层密度量化",
                "上部复合层/太古界风化壳/整体 P33 热图、井位图、网格与井汇总表",
                max_table_mb=50.0,
            ),
            Spec(
                "太古界/dfn_display_zflip_quanqu1",
                "附：ParaView Z 镜像展示包",
                "负 Z 展示口径；含 11 口井标注、manifest、预览图与 ParaView 标签脚本",
            ),
        ],
    ),
]


def iter_spec_files(spec: Spec) -> list[tuple[Path, int]]:
    """按 C 方案规则返回 spec 下要打包的文件。"""
    root = REPO / spec.src_rel
    if not root.is_dir():
        raise FileNotFoundError(f"来源目录不存在：{root}")
    picked: list[tuple[Path, int]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts[:-1]
        if any(part in SKIP_DIR_NAMES for part in rel_parts):
            continue
        suffix = path.suffix.lower()
        if suffix in ALWAYS_SKIP_SUFFIXES:
            continue
        size = path.stat().st_size
        if suffix in TABLE_SUFFIXES and size > spec.max_table_mb * MB:
            continue  # 大表明细，按 C 方案剔除
        picked.append((path, size))
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description="打包 C 方案交付集（砂砾岩矿区 v1 + 太古界 quanqu1）")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--archive-name", default=f"GAN-DFN_交付_C方案_砂砾岩矿区v1+太古界quanqu1_{PACKAGE_TAG}")
    parser.add_argument("--list-only", action="store_true", help="只列举文件与体积，不复制/打包")
    parser.add_argument("--keep-staging", action="store_true", help="保留中间 staging 目录")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    staging = out_dir / (args.archive_name + "_staging")
    doc_dir = staging / f"交付说明_{PACKAGE_TAG}"

    print(f"[打包] 项目根目录：{REPO}")
    print(f"[打包] 输出目录：{out_dir}")
    print("=" * 78)

    manifest_rows: list[dict[str, object]] = []
    total_bytes = 0

    for region in REGIONS:
        print(f"\n### {region.name}（{region.version}）")
        region_bytes = 0
        for spec in region.specs:
            files = iter_spec_files(spec)
            spec_bytes = sum(size for _, size in files)
            region_bytes += spec_bytes
            print(f"  {spec.step:34s} {len(files):5d} 文件  {spec_bytes / MB:9.1f} MB   {spec.src_rel}")
            if not args.list_only:
                for src, size in files:
                    rel = src.relative_to(REPO)
                    dst = staging / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    manifest_rows.append(
                        {
                            "工区": region.name,
                            "版本": region.version,
                            "步骤": spec.step,
                            "包内相对路径": str(rel),
                            "体积MB": round(size / MB, 3),
                        }
                    )
        total_bytes += region_bytes
        print(f"  小计：{region_bytes / MB:,.1f} MB")

    print("=" * 78)
    print(f"合计 {len(manifest_rows) if not args.list_only else '待打包'} 个文件，{total_bytes / MB:,.1f} MB（{total_bytes / 1024 ** 3:.2f} GB）")

    if args.list_only:
        return 0

    # ---------------------------------------------------------------- 说明与清单
    doc_dir.mkdir(parents=True, exist_ok=True)
    with open(doc_dir / "MANIFEST.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["工区", "版本", "步骤", "包内相对路径", "体积MB"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    readme = [
        "# GAN-DFN 交付包（C 方案：砂砾岩矿区 v1 + 太古界 quanqu1）",
        "",
        f"打包时间：{PACKAGE_TAG}",
        f"文件数：{len(manifest_rows)}，解压后体积：{total_bytes / MB:,.1f} MB（{total_bytes / 1024 ** 3:.2f} GB）",
        "",
        "## 怎么解压",
        "",
        "包内文件保留了**相对项目根目录**的层级（`优化阶段二/...`、`太古界/...`）。",
        "请把压缩包**解压到项目根目录**（即与 `优化阶段二/`、`太古界/` 同级的位置），",
        "文件会自动落回原位。换机器后绝对路径会变，项目内相对路径不变。",
        "",
        "```bash",
        f"tar -xzf {args.archive_name}.tar.gz -C <项目根目录>",
        "```",
        "",
        "## 包含范围（C 方案：交付浏览版）",
        "",
        "| 工区 | 步骤 | 内容 |",
        "|---|---|---|",
    ]
    for region in REGIONS:
        for i, spec in enumerate(region.specs):
            readme.append(f"| {region.name if i == 0 else ''} | {spec.step} | {spec.note} |")
    readme += [
        "",
        "## 不包含（按 C 方案剔除）",
        "",
        "- Step1–Step5 的井上样本表（真实井/成像监督组/虚拟井/统一训练样本）；",
        "- Step6 的密度体与先验体 `.sgy`（砂砾岩 30 GB、太古界 51 GB）；",
        "- Step7/Step8 的 `*_patches.csv` 单片明细与大审计表（保留汇总 JSON 与小 QC 表）；",
        "- `section_samples/*.npz` 剖面采样缓存。",
        "",
        "## 口径提醒",
        "",
        "1. **Z 约定**：正式文件 Z = |TWT|（ms，正）；`dfn_display_zflip*` 为负 Z 镜像展示版，仅供 ParaView 显示，不参与计算。",
        "2. **P33 成像标定**（砂砾岩 Step9b）：以车页1导眼为基准的层段常数因子，属「点标定全区」，空间分布由 DFN 决定；太古界未做标定。",
        "3. **地层口径**：砂砾岩 = 沙三段 / 沙四段；太古界 = 上部复合层（T4→T6）/ 太古界风化壳（T6→T7）。",
        "4. **坐标**：X=Easting、Y=Northing（米）；砂砾岩矿区 X 556150–581650 / Y 4193975–4212275；太古界 X 658887–679500 / Y 4231662–4247987。",
        "",
        "## 文件清单",
        "",
        "见同目录 `MANIFEST.csv`（逐文件：工区 / 版本 / 步骤 / 包内相对路径 / 体积）。",
    ]
    (doc_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- 打包
    archive_path = out_dir / f"{args.archive_name}.tar.gz"
    print(f"\n[打包] 生成 {archive_path} ...")
    with tarfile.open(archive_path, "w:gz") as tar:
        # 只打包顶层两项数据目录 + 交付说明，保证解压即归位
        for top in ("优化阶段二", "太古界", doc_dir.name):
            target = staging / top
            if target.exists():
                tar.add(target, arcname=top)
    size_gb = archive_path.stat().st_size / 1024 ** 3
    print(f"[打包] 完成：{archive_path}（{archive_path.stat().st_size / MB:,.1f} MB / {size_gb:.2f} GB）")

    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
        print("[打包] 已清理 staging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
