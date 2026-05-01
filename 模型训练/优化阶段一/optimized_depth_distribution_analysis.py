from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from fracture_prediction_compare_visualization import (
    build_interval_labels,
    build_regular_edges,
    convert_svg_to_png,
    draw_grouped_bar_panel,
    format_metric,
    histogram,
    mean_or_none,
    median_or_none,
    normalize_histogram,
    parse_float,
    quantile,
    read_csv_rows,
    svg_text,
    total_variation_distance,
    write_csv,
)


DEFAULT_PREDICTION_PATH = Path(
    # r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果/现有常规测井裂缝预测/车151HF/final_fracture_points.csv"
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果/现有常规测井裂缝预测/车页1导眼/final_fracture_points.csv"
)
DEFAULT_ACTUAL_PATH = Path(
    # r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/成像测井/裂缝提取/车151HF_fractures.csv"
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/成像测井/裂缝提取/车页1导眼_fractures.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/方案对比/研究内容一/第一轮优化深度分布分析"
)
# DEFAULT_DEPTH_MIN = 3655.0
# DEFAULT_DEPTH_MAX = 4935.0
DEFAULT_DEPTH_MIN = 3500.0
DEFAULT_DEPTH_MAX = 3750.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze first-round optimized fracture depth distribution against actual fractures."
    )
    parser.add_argument("--prediction-path", type=Path, default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--actual-path", type=Path, default=DEFAULT_ACTUAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bin-size", type=float, default=10.0)
    parser.add_argument("--depth-min", type=float, default=DEFAULT_DEPTH_MIN)
    parser.add_argument("--depth-max", type=float, default=DEFAULT_DEPTH_MAX)
    # parser.add_argument("--well-name", default="车151HF")
    parser.add_argument("--well-name", default="车页1导眼")
    return parser.parse_args()


def load_prediction_depths(path: Path) -> list[float]:
    rows = read_csv_rows(path)
    depths: list[float] = []
    for row in rows:
        depth = parse_float(row.get("TVD"))
        if depth is not None:
            depths.append(depth)
    return depths


def load_actual_depths(path: Path) -> list[float]:
    rows = read_csv_rows(path)
    depths: list[float] = []
    for row in rows:
        depth = parse_float(row.get("MD"))
        if depth is not None:
            depths.append(depth)
    return depths


def align_depth_window(
    prediction_depths: list[float],
    actual_depths: list[float],
    depth_min: float,
    depth_max: float,
) -> tuple[list[float], list[float], float, float]:
    if depth_max <= depth_min:
        raise ValueError("Depth range is invalid.")
    prediction_in_window = [depth for depth in prediction_depths if depth_min <= depth <= depth_max]
    actual_in_window = [depth for depth in actual_depths if depth_min <= depth <= depth_max]
    return prediction_in_window, actual_in_window, depth_min, depth_max


def compute_peak_label(labels: list[str], counts: list[int]) -> str:
    if not labels or not counts:
        return "NA"
    peak_idx = max(range(len(counts)), key=lambda idx: counts[idx])
    return f"{labels[peak_idx]} ({counts[peak_idx]})"


def create_depth_distribution_svg(
    output_svg: Path,
    well_name: str,
    labels: list[str],
    predicted_counts: list[int],
    actual_counts: list[int],
) -> None:
    width = 1500
    height = 900
    figure_elements: list[str] = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />',
        svg_text(40, 42, f"{well_name} 第一轮优化裂缝深度分布对比", font_size=26, font_weight="bold"),
    ]

    figure_elements.extend(
        draw_grouped_bar_panel(
            40,
            90,
            1420,
            760,
            "裂缝深度分布对比（横轴：深度范围，纵轴：裂缝数量）",
            labels=labels,
            series_a=actual_counts,
            series_b=predicted_counts,
            label_a="实际裂缝",
            label_b="第一轮优化预测",
            color_a="#4c78a8",
            color_b="#f58518",
            y_label="数量",
            normalize=False,
        )
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Microsoft YaHei, SimHei, Arial, sans-serif">'
        + "".join(figure_elements)
        + "</svg>"
    )
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bin_size <= 0:
        raise ValueError("--bin-size must be positive.")
    if args.depth_max <= args.depth_min:
        raise ValueError("--depth-max must be greater than --depth-min.")
    if not args.prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {args.prediction_path}")
    if not args.actual_path.exists():
        raise FileNotFoundError(f"Actual file not found: {args.actual_path}")

    prediction_depths = load_prediction_depths(args.prediction_path)
    actual_depths = load_actual_depths(args.actual_path)
    prediction_in_window, actual_in_window, depth_min, depth_max = align_depth_window(
        prediction_depths,
        actual_depths,
        args.depth_min,
        args.depth_max,
    )

    edges = build_regular_edges(depth_min, depth_max, args.bin_size, include_endpoint=True)
    labels = build_interval_labels(edges, integer_format=True)

    actual_counts = histogram(actual_in_window, edges)
    predicted_counts = histogram(prediction_in_window, edges)
    actual_probs = normalize_histogram(actual_counts)
    predicted_probs = normalize_histogram(predicted_counts)
    bin_abs_diffs = [abs(pred - act) for pred, act in zip(predicted_counts, actual_counts)]

    summary = {
        "well_name": args.well_name,
        "depth_axis_policy": "PredictionTVD_And_ActualMDTreatedAsUnifiedTVD",
        "depth_window_min": round(depth_min, 6),
        "depth_window_max": round(depth_max, 6),
        "bin_size": float(args.bin_size),
        "prediction_total_count": len(prediction_depths),
        "prediction_in_window_count": len(prediction_in_window),
        "actual_total_count": len(actual_depths),
        "actual_in_window_count": len(actual_in_window),
        "count_ratio": (len(prediction_in_window) / len(actual_in_window)) if actual_in_window else None,
        "pred_mean": mean_or_none(prediction_in_window),
        "pred_median": median_or_none(prediction_in_window),
        "act_mean": mean_or_none(actual_in_window),
        "act_median": median_or_none(actual_in_window),
        "pred_peak_label": compute_peak_label(labels, predicted_counts),
        "act_peak_label": compute_peak_label(labels, actual_counts),
        "tv_distance": total_variation_distance(predicted_probs, actual_probs),
        "bin_abs_diff_median": median_or_none(bin_abs_diffs),
        "bin_abs_diff_p90": quantile(bin_abs_diffs, 0.90),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bins_csv = output_dir / "che151hf_first_round_depth_distribution_bins.csv"
    summary_json = output_dir / "che151hf_first_round_depth_distribution_summary.json"
    svg_path = output_dir / "che151hf_first_round_depth_distribution_tmp.svg"
    png_path = output_dir / "che151hf_first_round_depth_distribution.png"

    rows = []
    for idx, label in enumerate(labels):
        rows.append(
            {
                "DepthBinLabel": label,
                "DepthBinStart": edges[idx],
                "DepthBinEnd": edges[idx + 1],
                "ActualCount": actual_counts[idx],
                "PredictedCount": predicted_counts[idx],
                "CountDifference": predicted_counts[idx] - actual_counts[idx],
                "ActualProportion": actual_probs[idx],
                "PredictedProportion": predicted_probs[idx],
            }
        )
    write_csv(bins_csv, rows)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    create_depth_distribution_svg(
        output_svg=svg_path,
        well_name=args.well_name,
        labels=labels,
        predicted_counts=predicted_counts,
        actual_counts=actual_counts,
    )
    convert_svg_to_png(svg_path, png_path)
    if svg_path.exists():
        svg_path.unlink()

    print(f"Output directory: {output_dir}")
    print(f"PNG: {png_path}")
    print(f"Bins CSV: {bins_csv}")
    print(f"Summary JSON: {summary_json}")
    print(
        f"[{args.well_name}] prediction_in_window={len(prediction_in_window)}/{len(prediction_depths)}, "
        f"actual={len(actual_depths)}, tv_distance={format_metric(summary['tv_distance'])}"
    )


if __name__ == "__main__":
    main()
