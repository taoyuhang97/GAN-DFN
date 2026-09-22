#!/usr/bin/env python3
"""G1 闸门：比对 Step3 md1 与 v5 的成像标签，确认"除 405 外逐行一致"。

用法::

    python3 太古界/tools/compare_md1_vs_v5_step3.py \
        --md1 太古界/step3_imaging_groups/output/taigu_step3_imaging_md1 \
        --v5  太古界/step3_imaging_groups/output/taigu_step3_imaging_v5 \
        --out 太古界/output_md1_logs/gate_g1_step3.json

v5 的 405 标签贴在 MD 4104.7–4557.8（错误井段），md1 贴在 3676.6–3998.0，
两者行键不同，故 405 只报告行数与 MD 区间（预期差异），不参与"逐行一致"判定。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPARE_COLUMNS = ["Density", "GT_POINT_FLAG", "RawPointCount", "FracDip", "FracAzimuth", "StrataName"]
EXPECTED_DIFF_WELLS = {"埕北古斜405"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Step3 md1 vs v5 imaging labels.")
    parser.add_argument("--md1", required=True)
    parser.add_argument("--v5", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--atol", type=float, default=1.0e-6)
    return parser.parse_args()


def read_well(root: Path, well: str) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "groups").glob(f"{well}*.csv")):
        frames.append(pd.read_csv(path, encoding="utf-8-sig"))
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["_key"] = np.round(pd.to_numeric(frame["MD"], errors="coerce"), 3)
    return frame.dropna(subset=["_key"]).sort_values("_key").reset_index(drop=True)


def compare_well(md1: pd.DataFrame, v5: pd.DataFrame, atol: float) -> dict[str, object]:
    out: dict[str, object] = {
        "md1_rows": int(len(md1)),
        "v5_rows": int(len(v5)),
        "md1_md_range": [float(md1["MD"].min()), float(md1["MD"].max())] if len(md1) else None,
        "v5_md_range": [float(v5["MD"].min()), float(v5["MD"].max())] if len(v5) else None,
    }
    if not len(md1) or not len(v5):
        out["status"] = "rows_missing"
        return out
    left = md1.drop_duplicates("_key").set_index("_key")
    right = v5.drop_duplicates("_key").set_index("_key")
    common = left.index.intersection(right.index)
    only_md1 = left.index.difference(right.index)
    only_v5 = right.index.difference(left.index)
    out["only_in_md1_rows"] = int(len(only_md1))
    out["only_in_v5_rows"] = int(len(only_v5))
    out["only_in_md1_md_range"] = [float(only_md1.min()), float(only_md1.max())] if len(only_md1) else None
    out["only_in_v5_md_range"] = [float(only_v5.min()), float(only_v5.max())] if len(only_v5) else None
    diffs: dict[str, dict[str, float]] = {}
    for column in COMPARE_COLUMNS:
        if column not in left.columns or column not in right.columns:
            diffs[column] = {"max_abs_diff": float("nan"), "differing_rows": -1}
            continue
        a = left.loc[common, column]
        b = right.loc[common, column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            diff = (pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")).abs()
            diffs[column] = {
                "max_abs_diff": float(np.nanmax(diff)) if len(diff) and np.isfinite(diff).any() else 0.0,
                "differing_rows": int((diff.fillna(0.0) > atol).sum()),
            }
        else:
            same = a.astype(str).eq(b.astype(str))
            diffs[column] = {"max_abs_diff": float("nan"), "differing_rows": int((~same).sum())}
    out["column_diffs"] = diffs
    out["identical"] = bool(
        len(only_md1) == 0 and len(only_v5) == 0 and all(int(v["differing_rows"]) == 0 for v in diffs.values())
    )
    return out


def main() -> int:
    args = parse_args()
    md1_root = Path(args.md1).resolve()
    v5_root = Path(args.v5).resolve()
    wells_md1 = {p.name.split("_g0")[0] for p in (md1_root / "groups").glob("*.csv")}
    wells_v5 = {p.name.split("_g0")[0] for p in (v5_root / "groups").glob("*.csv")}
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
        detail = compare_well(read_well(md1_root, well), read_well(v5_root, well), args.atol)
        report["wells"][well] = detail
        if well in EXPECTED_DIFF_WELLS:
            continue
        if not detail.get("identical", False):
            unexpected.append(well)
    report["unexpected_diff_wells"] = unexpected
    report["status"] = "pass" if not unexpected and not report["wells_md1_only"] and not report["wells_v5_only"] else "fail"
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
