#!/usr/bin/env python3
"""Build the full attribute-grid horizon contract and its window cache."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from build_horizon_contract import main as build_contract
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--attribute-header-csv", type=Path, required=True)
    p.add_argument("--horizon-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--replace-output", action="store_true")
    a = p.parse_args()
    staging = a.output_dir / "_attribute_grid_input.csv"
    df = pd.read_csv(a.attribute_header_csv, encoding="utf-8-sig")
    df = df.sort_values("TraceIdx").reset_index(drop=True)
    if len(df) != 1650 * 1307:
        raise ValueError(f"expected 2,156,550 attribute traces, got {len(df)}")
    df["IX"] = (df.index // 1307).astype("int32")
    df["IY"] = (df.index % 1307).astype("int32")
    staging.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(staging, index=False, encoding="utf-8-sig")
    old = sys.argv
    sys.argv = ["build_horizon_contract", "--horizon-dir", str(a.horizon_dir),
                "--demo-grid-csv", str(staging), "--output-dir", str(a.output_dir)]
    if a.replace_output:
        sys.argv.append("--replace-output")
    try:
        return build_contract()
    finally:
        sys.argv = old


if __name__ == "__main__":
    raise SystemExit(main())
