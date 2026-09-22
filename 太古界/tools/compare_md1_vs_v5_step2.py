#!/usr/bin/env python3
"""G1 闸门：比对 Step2 md1 与 v5 的段表，确认"除 405 外逐行一致"。

用法::

    python3 太古界/tools/compare_md1_vs_v5_step2.py \
        --md1 太古界/step2_well_log_segments/output/taigu_step2_regular_md1 \
        --v5  太古界/step2_well_log_segments/output/taigu_step2_regular_v5 \
        --out 太古界/output_md1_logs/gate_g1_step2.json

判据：以 ``(WellName, round(MD, 3))`` 为键做外连接；对同名井统计
只在一边出现的行数，以及在两边都有的行上各比较列的最大绝对差。
预期只有 **埕北古斜405** 出现差异（其余井必须 0 差异）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPARE_COLUMNS = ["TVD", "X", "Y", "TIME", "InImagingInterval", "InHorizonLayer", "UseCase"]
EXPECTED_DIFF_WELLS = {"埕北古斜405"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Step2 md1 vs v5 segment tables.")
    parser.add_argument("--md1", required=True)
    parser.add_argument("--v5", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    return parser.parse_args()


def list_wells(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def read_well(root: Path, well: str) -> pd.DataFrame:
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in sorted((root / well).glob("*.csv"))]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["_key"] = np.round(pd.to_numeric(frame["MD"], errors="coerce"), 3)
    return frame.dropna(subset=["_key"]).sort_values("_key").reset_index(drop=True)


def compare_well(md1: pd.DataFrame, v5: pd.DataFrame, rtol: float, atol: float) -> dict[str, object]:
    result: dict[str, object] = {
        "md1_rows": int(len(md1)),
        "v5_rows": int(len(v5)),
        "md1_md_range": [float(md1["MD"].min()), float(md1["MD"].max())] if len(md1) else None,
        "v5_md_range": [float(v5["MD"].min()), float(v5["MD"].max())] if len(v5) else None,
    }
    if not len(md1) or not len(v5):
        result["status"] = "rows_missing"
        return result
    left = md1.drop_duplicates("_key").set_index("_key")
    right = v5.drop_duplicates("_key").set_index("_key")
    only_md1 = left.index.difference(right.index)
    only_v5 = right.index.difference(left.index)
    common = left.index.intersection(right.index)
    result["only_in_md1_rows"] = int(len(only_md1))
    result["only_in_v5_rows"] = int(len(only_v5))
    result["only_in_md1_md_range"] = (
        [float(only_md1.min()), float(only_md1.max())] if len(only_md1) else None
    )
    result["only_in_v5_md_range"] = [float(only_v5.min()), float(only_v5.max())] if len(only_v5) else None
    column_diffs: dict[str, dict[str, float]] = {}
    for column in COMPARE_COLUMNS:
        if column not in left.columns or column not in right.columns:
            column_diffs[column] = {"max_abs_diff": float("nan"), "differing_rows": -1}
            continue
        a = left.loc[common, column]
        b = right.loc[common, column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            diff = (pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")).abs()
            differing = int((diff > atol).sum())
            column_diffs[column] = {"max_abs_diff": float(np.nanmax(diff)) if len(diff) else 0.0, "differing_rows": differing}
        else:
            same = a.astype(str).eq(b.astype(str))
            column_diffs[column] = {"max_abs_diff": float("nan"), "differing_rows": int((~same).sum())}
    result["column_diffs"] = column_diffs
    result["identical"] = bool(
        len(only_md1) == 0
        and len(only_v5) == 0
        and all(int(v["differing_rows"]) == 0 for v in column_diffs.values())
    )
    return result


def main() -> int:
    args = parse_args()
    md1_root = Path(args.md1).resolve()
    v5_root = Path(args.v5).resolve()
    wells_md1 = set(list_wells(md1_root))
    wells_v5 = set(list_wells(v5_root))
    report: dict[str, object] = {
        "md1_root": str(md1_root),
        "v5_root": str(v5_root),
        "wells_md1_only": sorted(wells_md1 - wells_v5),
        "wells_v5_only": sorted(wells_v5 - wells_md1),
        "expected_diff_wells": sorted(EXPECTED_DIFF_WELLS),
        "wells": {},
    }
    unexpected: list[str] = []
    for well in sorted(wells_md1 & wells_v5):
        detail = compare_well(read_well(md1_root, well), read_well(v5_root, well), args.rtol, args.atol)
        report["wells"][well] = detail
        if well in EXPECTED_DIFF_WELLS:
            continue
        if not detail.get("identical", False):
            unexpected.append(well)
    report["unexpected_diff_wells"] = unexpected
    report["status"] = "pass" if not unexpected and not report["wells_md1_only"] and not report["wells_v5_only"] else "fail"
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
