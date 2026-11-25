#!/usr/bin/env python3
"""Predict the 2 ms base-below supplementary density on the 5x5 km pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TAIGU = REPO / "太古界"
for path in (TAIGU, TAIGU / "supp_layer", TAIGU / "base_supp_layer"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from base_supp_framework import load_base_supp_framework  # noqa: E402
from supp_layer import predict_supp_density as predictor  # noqa: E402


OUTPUT_ROOT = TAIGU / "base_supp_layer/output/taigu_base_supp_5km_v1"


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    own_args, predictor_args = parser.parse_known_args()
    output_root = own_args.output_root.resolve()
    sys.argv = [sys.argv[0], *predictor_args]

    predictor.load_supp_framework = load_base_supp_framework
    predictor.LAYER_NAME = "基底以下补充层"
    predictor.MODEL_LAYER = "太古界风化壳"
    predictor.LAYER_CODE = 4.0
    predictor.DEFAULT_CONTRACT = output_root / "base_supp_horizon_contract.npy"
    predictor.DEFAULT_BLOCK = output_root / "base_supp_horizon_contract_block.csv"
    predictor.DEFAULT_OUTPUT = output_root / "step6a_density"
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--contract",
                str(predictor.DEFAULT_CONTRACT),
                "--block-csv",
                str(predictor.DEFAULT_BLOCK),
                "--output-dir",
                str(predictor.DEFAULT_OUTPUT),
            ]
        )
    return predictor.main()


if __name__ == "__main__":
    raise SystemExit(main())
