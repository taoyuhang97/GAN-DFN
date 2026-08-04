from __future__ import annotations

import argparse
import json

from step4_relation_demo import DemoConfig, run_relation_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="第四步最小 demo：按层位统计属性-裂缝密度关系。"
    )
    parser.add_argument("--input-csv", required=True, help="统一样本表 csv 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--layer-col", default="LayerGroup", help="层位列名")
    parser.add_argument("--target-col", default="FractureDensity", help="裂缝密度列名")
    parser.add_argument("--weight-col", default="PointConfidence", help="样点权重列名")
    parser.add_argument(
        "--attribute-cols",
        nargs="*",
        default=None,
        help="手动指定属性列；若不传，则自动识别数值型属性列",
    )
    parser.add_argument("--bins", type=int, default=10, help="分箱数，默认 10")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="每个层位-属性组合的最小有效样本数",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="计算设备，默认 auto",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help="只处理指定层位；若不传，则处理输入表中全部层位",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = DemoConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        layer_col=args.layer_col,
        target_col=args.target_col,
        weight_col=args.weight_col,
        attribute_cols=args.attribute_cols,
        bins=args.bins,
        min_samples=args.min_samples,
        device=args.device,
        layers=args.layers,
    )
    result = run_relation_demo(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
