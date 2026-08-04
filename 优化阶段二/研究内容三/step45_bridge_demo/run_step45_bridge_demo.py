from __future__ import annotations

import argparse
import json

from step45_bridge_demo import BridgeConfig, run_step45_bridge_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="第四步到第五步桥接 demo：将关系表映射为局部裂缝发育控制场。"
    )
    parser.add_argument("--sample-csv", required=True, help="第三步统一样本表路径")
    parser.add_argument("--relation-bins-csv", required=True, help="第四步 relation_bins.csv 路径")
    parser.add_argument(
        "--relation-summary-csv",
        required=True,
        help="第四步 relation_summary.csv 路径",
    )
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--layer-col", default="LayerGroup", help="层位列名")
    parser.add_argument("--x-col", default="X", help="X 坐标列名")
    parser.add_argument("--y-col", default="Y", help="Y 坐标列名")
    parser.add_argument("--time-col", default="TIME", help="时间列名")
    parser.add_argument("--target-col", default="FractureDensity", help="真实裂缝密度列名")
    parser.add_argument(
        "--weak-target-col",
        default="FractureDensityWeak",
        help="第三步输出的弱标签密度列名",
    )
    parser.add_argument(
        "--point-conf-col",
        default="PointConfidence",
        help="样点级可信度列名",
    )
    parser.add_argument(
        "--well-conf-col",
        default="WellConfidence",
        help="井级可信度列名",
    )
    parser.add_argument(
        "--min-attr-weight",
        type=float,
        default=0.05,
        help="保留属性关系所需的最小绝对相关权重",
    )
    parser.add_argument(
        "--min-valid-attributes",
        type=int,
        default=1,
        help="每个样点最少需要多少个有效属性关系才能输出控制分数",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="计算设备，默认 auto",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = BridgeConfig(
        sample_csv=args.sample_csv,
        relation_bins_csv=args.relation_bins_csv,
        relation_summary_csv=args.relation_summary_csv,
        output_dir=args.output_dir,
        layer_col=args.layer_col,
        x_col=args.x_col,
        y_col=args.y_col,
        time_col=args.time_col,
        target_col=args.target_col,
        weak_target_col=args.weak_target_col,
        point_conf_col=args.point_conf_col,
        well_conf_col=args.well_conf_col,
        min_attr_weight=args.min_attr_weight,
        min_valid_attributes=args.min_valid_attributes,
        device=args.device,
    )
    result = run_step45_bridge_demo(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
