# -*- coding: utf-8 -*-
"""Plot Step5 source-well and virtual-well locations in the mine block."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sys

SECTION_DIR = Path(__file__).resolve().parent.parent / "step9_section_visualize"
if str(SECTION_DIR) not in sys.path:
    sys.path.insert(0, str(SECTION_DIR))
from build_all_area_section_visualization import configure_matplotlib_fonts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-block", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    configure_matplotlib_fonts()

    index = pd.read_csv(args.index, encoding="utf-8-sig")
    source = index.drop_duplicates("SourceWellName").copy()
    virtual = index.drop_duplicates("VirtualWellName").copy()
    block = json.loads(args.target_block.read_text(encoding="utf-8"))
    block = block.get("target_block", block)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(virtual["VirtualAnchorX"], virtual["VirtualAnchorY"], s=10,
               c="#7C3AED", alpha=0.28, linewidths=0, label=f"虚拟井（{len(virtual)}口）", zorder=2)
    ax.scatter(source["SourceAnchorX"], source["SourceAnchorY"], s=42,
               c="#FFD400", edgecolors="#111827", linewidths=0.7,
               label=f"真实测井源井（{len(source)}口）", zorder=4)
    for _, row in source.iterrows():
        ax.annotate(str(row["SourceWellName"]), (row["SourceAnchorX"], row["SourceAnchorY"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7,
                    color="#111827", zorder=5)
    ax.set_xlim(float(block["x_min"]), float(block["x_max"]))
    ax.set_ylim(float(block["y_min"]), float(block["y_max"]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("Step5 真实测井源井与虚拟测井位置分布")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(loc="best", framealpha=0.92)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    summary = {"status": "pass", "source_well_count": int(len(source)),
               "virtual_well_count": int(len(virtual)),
               "virtual_tracks_per_source": int(index.groupby("SourceWellName").size().mode().iloc[0]),
               "index_path": str(args.index.resolve()), "output": str(args.output.resolve())}
    args.output.with_name(args.output.stem + "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
