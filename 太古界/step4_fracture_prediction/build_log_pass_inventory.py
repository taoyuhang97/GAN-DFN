"""Build a reproducible LAS-pass and resistivity-pair inventory.

This command is intentionally read-only with respect to raw logs.  It records
which exact LAS pass supports each curve pair before any standardization or
interpolation is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import lasio
import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = CURRENT_DIR.parent
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))

from common.well_curve_catalog import curve_catalog_rows, find_resistivity_pairs, pair_catalog_rows  # noqa: E402


INVALID_SENTINELS = (-999.25, -9999.0, -99999.0, 9999.0, 99999.0)
TARGET_INTERVALS = {
    "车页1导眼": (3500.0, 3750.0),
    "车662": (3475.0, 3973.0),
    "车663": (3879.0, 4281.0),
}
DEFAULT_ROOTS = (
    Path("/data/shared/project-oil/wx数据/砂砾岩/测井"),
    Path("/data/shared/project-oil/wx数据/砂砾岩/成像测井-测井曲线"),
    Path("/data/shared/project-oil/wx数据/砂砾岩/测井数据补充-20260723"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory LAS passes and verified resistivity pairs.")
    parser.add_argument("--output-dir", type=Path, default=CURRENT_DIR / "output/log_pass_inventory_md_v2")
    parser.add_argument("--roots", nargs="*", type=Path, default=list(DEFAULT_ROOTS))
    return parser.parse_args()


def canonical_well_name(path: Path) -> str:
    text = path.stem.replace("_", "")
    if "车页1" in text:
        return "车页1导眼"
    match = re.search(r"车(?:斜)?\d+(?:-\d+)?", text)
    return match.group(0) if match else path.stem.split("@")[0]


def valid_resistivity(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    for sentinel in INVALID_SENTINELS:
        out = out.mask(np.isclose(out, sentinel, equal_nan=False))
    return out.mask((out <= 0.0) | (out >= 9000.0))


def valid_curve(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    for sentinel in INVALID_SENTINELS:
        out = out.mask(np.isclose(out, sentinel, equal_nan=False))
    return out


def load_las_frame(path: Path) -> tuple[pd.DataFrame, object]:
    las = lasio.read(str(path), ignore_header_errors=True)
    frame = las.df().reset_index()
    frame = frame.rename(columns={frame.columns[0]: "MD"})
    frame.columns = [str(column).upper().strip() for column in frame.columns]
    frame["MD"] = pd.to_numeric(frame["MD"], errors="coerce")
    return frame.dropna(subset=["MD"]).sort_values("MD").reset_index(drop=True), las


def curve_units(las: object) -> dict[str, str]:
    return {str(curve.mnemonic).upper().strip(): str(curve.unit or "").strip() for curve in las.curves}


def pass_row(path: Path, root: Path, frame: pd.DataFrame, las: object) -> dict[str, object]:
    md = frame["MD"]
    spacing = md.diff().dropna()
    return {
        "WellName": canonical_well_name(path),
        "SourceRoot": str(root),
        "SourcePath": str(path),
        "ContentSHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "FileName": path.name,
        "MDMin": float(md.min()),
        "MDMax": float(md.max()),
        "RowCount": int(len(frame)),
        "MedianStepM": float(spacing.median()) if not spacing.empty else np.nan,
        "LasNull": str(getattr(getattr(las, "well", {}), "NULL", {}).value) if hasattr(getattr(las, "well", {}), "NULL") else "",
        "CurveCount": int(len(frame.columns) - 1),
        "Curves": ";".join(column for column in frame.columns if column != "MD"),
    }


def pair_rows(pass_info: dict[str, object], frame: pd.DataFrame, units: dict[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    interval = TARGET_INTERVALS.get(str(pass_info["WellName"]))
    for definition in find_resistivity_pairs(set(frame.columns)):
        deep = valid_resistivity(frame[definition.deep_mnemonic])
        near = valid_resistivity(frame[definition.near_mnemonic])
        pair_valid = deep.notna() & near.notna()
        row = {
            **pass_info,
            "PairType": definition.pair_type,
            "MeasurementFamily": definition.measurement_family,
            "PairChineseName": definition.chinese_name,
            "DeepCurve": definition.deep_mnemonic,
            "NearCurve": definition.near_mnemonic,
            "DeepUnit": units.get(definition.deep_mnemonic, ""),
            "NearUnit": units.get(definition.near_mnemonic, ""),
            "PairValidRows": int(pair_valid.sum()),
            "PairValidCoverage": float(pair_valid.mean()),
        }
        if interval is not None:
            target = frame["MD"].between(*interval)
            target_pair = pair_valid & target
            row.update(
                {
                    "TargetMDMin": float(interval[0]),
                    "TargetMDMax": float(interval[1]),
                    "TargetRows": int(target.sum()),
                    "TargetPairValidRows": int(target_pair.sum()),
                    "TargetPairCoverage": float(target_pair.sum() / target.sum()) if target.any() else 0.0,
                }
            )
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pass_rows: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    seen: set[Path] = set()
    for root in args.roots:
        if not root.exists():
            errors.append({"SourceRoot": str(root), "SourcePath": "", "Error": "missing_root"})
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".las", ".lis"} or path in seen:
                continue
            seen.add(path)
            try:
                frame, las = load_las_frame(path)
                info = pass_row(path, root, frame, las)
                pass_rows.append(info)
                pairs.extend(pair_rows(info, frame, curve_units(las)))
            except Exception as exc:  # inventory should record malformed legacy files rather than stop
                errors.append({"SourceRoot": str(root), "SourcePath": str(path), "Error": str(exc)})

    pd.DataFrame(curve_catalog_rows()).to_csv(args.output_dir / "curve_catalog.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pair_catalog_rows()).to_csv(args.output_dir / "resistivity_pair_catalog.csv", index=False, encoding="utf-8-sig")
    pass_df = pd.DataFrame(pass_rows)
    pair_df = pd.DataFrame(pairs)
    if not pass_df.empty:
        pass_df["DuplicateContentCount"] = pass_df.groupby("ContentSHA256")["ContentSHA256"].transform("size")
        pass_df["IsDuplicateContentCopy"] = pass_df.duplicated("ContentSHA256", keep="first")
    if not pair_df.empty:
        pair_df["DuplicateContentCount"] = pair_df.groupby("ContentSHA256")["ContentSHA256"].transform("size")
        pair_df["IsDuplicateContentCopy"] = pair_df.duplicated(["ContentSHA256", "PairType"], keep="first")
    pass_df.sort_values(["WellName", "MDMin", "SourcePath"]).to_csv(args.output_dir / "log_pass_inventory.csv", index=False, encoding="utf-8-sig")
    pair_df.sort_values(["WellName", "PairType", "MDMin", "SourcePath"]).to_csv(args.output_dir / "resistivity_pair_inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["SourceRoot", "SourcePath", "Error"]).to_csv(
        args.output_dir / "inventory_read_errors.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "source_workbook": str(FORMAL_ROOT / "docs/参考资料/测井曲线名称与含义对照表.xls"),
        "las_file_count": len(pass_rows),
        "unique_las_content_count": int(pass_df["ContentSHA256"].nunique()) if not pass_df.empty else 0,
        "pair_pass_count": len(pairs),
        "read_error_count": len(errors),
        "target_well_pair_passes": [row for row in pairs if row["WellName"] in TARGET_INTERVALS],
    }
    (args.output_dir / "inventory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("las_file_count", "pair_pass_count", "read_error_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
