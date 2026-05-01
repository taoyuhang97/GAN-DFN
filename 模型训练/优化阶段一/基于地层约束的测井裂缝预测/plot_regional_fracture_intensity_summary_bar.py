from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果/层位裂缝发育情况统计"
)
DEFAULT_INPUT_FILENAME = "regional_strata_fracture_intensity_summary.csv"
DEFAULT_OUTPUT_DIRNAME = "区域层位裂缝发育柱状图"
DEFAULT_OUTPUT_FILENAME = "regional_fracture_intensity_summary_bar.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_summary_frame(input_root: Path) -> pd.DataFrame:
    csv_path = input_root / DEFAULT_INPUT_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到区域层位统计文件: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required_cols = ["top_surface", "base_surface", "fracture_density_per_100m"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"文件缺少必要列: {missing_cols}")

    out = df.copy()
    out["top_surface"] = out["top_surface"].astype(str).str.strip()
    out["base_surface"] = out["base_surface"].astype(str).str.strip()
    out["fracture_density_per_100m"] = pd.to_numeric(
        out["fracture_density_per_100m"], errors="coerce"
    )
    out = out[
        out["top_surface"].ne("")
        & out["base_surface"].ne("")
        & out["fracture_density_per_100m"].notna()
    ].copy()
    if out.empty:
        raise ValueError("区域层位统计文件中没有可绘图的数据")

    out["surface_pair"] = out["top_surface"] + "-" + out["base_surface"]
    return out.reset_index(drop=True)


def plot_bar_figure(df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    configure_matplotlib()

    x_labels = df["surface_pair"].tolist()
    y_values = df["fracture_density_per_100m"].tolist()

    fig_width = max(10.0, len(x_labels) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    bars = ax.bar(
        x_labels,
        y_values,
        color="#4C78A8",
        edgecolor="#2F4B6E",
        linewidth=1.0,
    )

    max_value = max(y_values) if y_values else 0.0
    offset = max(max_value * 0.02, 0.1)
    for bar, value in zip(bars, y_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("地层界面组合", fontsize=12)
    ax.set_ylabel("裂缝密度/(条/100m)", fontsize=12)
    ax.set_title("区域层位裂缝发育强弱对比", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=35)
    ax.margins(x=0.02)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_root}")

    output_dir = Path(args.output_dir) if args.output_dir else input_root / DEFAULT_OUTPUT_DIRNAME
    df = load_summary_frame(input_root=input_root)
    output_path = output_dir / DEFAULT_OUTPUT_FILENAME
    plot_bar_figure(df=df, output_path=output_path, dpi=int(args.dpi))
    print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
