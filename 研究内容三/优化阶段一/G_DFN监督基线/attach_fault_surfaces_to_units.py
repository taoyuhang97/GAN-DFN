# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from baseline_common import DEFAULT_DOCX_PATH, append_lines_to_docx, write_csv_utf8, write_json


UNIT_ID_PATTERN = re.compile(r"^BX(?P<block_x>\d+)_BY(?P<block_y>\d+)$", flags=re.IGNORECASE)
FAULT_PATCH_PATTERN = re.compile(r"^(?P<fault_name>.+)__i(?P<cell_i>\d+)_j(?P<cell_j>\d+)$", flags=re.IGNORECASE)

DEFAULT_FAULT_PATCHES_ROOT = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容二\单元实验\fault_patches_out\patches"
)
DEFAULT_FAULT_SUMMARY_CSV = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容二\单元实验\fault_patches_out\fault_patches_summary.csv"
)

MANIFEST_COLUMNS = [
    "UnitID",
    "BlockX",
    "BlockY",
    "FaultName",
    "FaultPatchFile",
    "FaultGroupDir",
    "FaultPatchSourcePath",
    "FaultPatchTargetPath",
    "CellI",
    "CellJ",
    "Area3D",
    "CenterX",
    "CenterY",
    "CenterTIME",
    "BBoxXMin",
    "BBoxXMax",
    "BBoxYMin",
    "BBoxYMax",
    "BBoxZMin",
    "BBoxZMax",
    "StrikeDeg",
    "DipDeg",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attach per-unit fault surface VTP patches to generated unit DFN directories.")
    parser.add_argument("--units-root", type=Path, required=True, help="Path to a generated run's units directory.")
    parser.add_argument("--fault-patches-root", type=Path, default=DEFAULT_FAULT_PATCHES_ROOT)
    parser.add_argument("--fault-summary-csv", type=Path, default=DEFAULT_FAULT_SUMMARY_CSV)
    parser.add_argument("--summary-dir", type=Path, help="Optional aggregated output directory. Defaults to <units-root>/../aggregated")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--unit-id", nargs="+")
    parser.add_argument("--unit-id-csv", type=Path)
    parser.add_argument("--block-x-start", type=int)
    parser.add_argument("--block-x-end", type=int)
    parser.add_argument("--block-y-start", type=int)
    parser.add_argument("--block-y-end", type=int)
    parser.add_argument("--limit-units", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean-target", action="store_true")
    parser.add_argument("--run-name", type=str, default=f"fault_surface_attach_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return parser


def parse_unit_id(unit_id: str) -> tuple[int, int]:
    matched = UNIT_ID_PATTERN.match(str(unit_id).strip())
    if not matched:
        raise ValueError(f"invalid UnitID: {unit_id}")
    return int(matched.group("block_x")), int(matched.group("block_y"))


def load_unit_ids_from_csv(unit_id_csv: Path) -> list[str]:
    df = pd.read_csv(unit_id_csv, encoding="utf-8-sig")
    if "UnitID" in df.columns:
        return df["UnitID"].dropna().astype(str).tolist()
    if {"BlockX", "BlockY"}.issubset(df.columns):
        return [f"BX{int(block_x)}_BY{int(block_y)}" for block_x, block_y in df[["BlockX", "BlockY"]].dropna().to_numpy()]
    if df.shape[1] >= 1:
        return df.iloc[:, 0].dropna().astype(str).tolist()
    return []


def discover_unit_ids(units_root: Path) -> list[str]:
    if not units_root.exists():
        raise FileNotFoundError(f"units_root not found: {units_root}")
    unit_ids = [
        path.name
        for path in sorted(units_root.iterdir())
        if path.is_dir() and UNIT_ID_PATTERN.match(path.name)
    ]
    return sorted(unit_ids, key=parse_unit_id)


def collect_selected_unit_ids(args: argparse.Namespace, units_root: Path) -> list[str]:
    unit_ids: set[str] = set()
    if args.unit_id:
        unit_ids.update(str(unit_id).strip() for unit_id in args.unit_id if str(unit_id).strip())
    if args.unit_id_csv:
        unit_ids.update(load_unit_ids_from_csv(Path(args.unit_id_csv)))
    if None not in (args.block_x_start, args.block_x_end, args.block_y_start, args.block_y_end):
        for block_x in range(int(args.block_x_start), int(args.block_x_end) + 1):
            for block_y in range(int(args.block_y_start), int(args.block_y_end) + 1):
                unit_ids.add(f"BX{block_x}_BY{block_y}")
    if not unit_ids:
        unit_ids.update(discover_unit_ids(units_root))
    ordered = sorted(unit_ids, key=parse_unit_id)
    if args.limit_units:
        ordered = ordered[: int(args.limit_units)]
    return ordered


def load_fault_summary_df(fault_summary_csv: Path) -> pd.DataFrame:
    if not fault_summary_csv.exists():
        raise FileNotFoundError(f"fault_summary.csv not found: {fault_summary_csv}")
    df = pd.read_csv(fault_summary_csv, encoding="utf-8-sig")
    if df.empty:
        return df
    numeric_cols = [
        "cell_i",
        "cell_j",
        "area_3d",
        "cx",
        "cy",
        "cz",
        "bbox_xmin",
        "bbox_xmax",
        "bbox_ymin",
        "bbox_ymax",
        "bbox_zmin",
        "bbox_zmax",
        "strike_deg",
        "dip_deg",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "fault_name" in df.columns:
        df["fault_name"] = df["fault_name"].astype(str)
    return df


def build_fault_summary_lookup(summary_df: pd.DataFrame) -> dict[tuple[str, int, int], dict[str, Any]]:
    lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    if summary_df.empty:
        return lookup
    for row in summary_df.to_dict("records"):
        fault_name = str(row.get("fault_name", ""))
        cell_i = pd.to_numeric(row.get("cell_i"), errors="coerce")
        cell_j = pd.to_numeric(row.get("cell_j"), errors="coerce")
        if pd.isna(cell_i) or pd.isna(cell_j):
            continue
        key = (fault_name, int(cell_i), int(cell_j))
        if key not in lookup:
            lookup[key] = row
    return lookup


def index_fault_patch_files(fault_patches_root: Path) -> dict[tuple[int, int], list[dict[str, Any]]]:
    if not fault_patches_root.exists():
        raise FileNotFoundError(f"fault_patches_root not found: {fault_patches_root}")
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for path in sorted(fault_patches_root.rglob("*.vtp")):
        matched = FAULT_PATCH_PATTERN.match(path.stem)
        if not matched:
            continue
        cell_i = int(matched.group("cell_i"))
        cell_j = int(matched.group("cell_j"))
        grouped.setdefault((cell_i, cell_j), []).append(
            {
                "fault_name": matched.group("fault_name"),
                "cell_i": cell_i,
                "cell_j": cell_j,
                "path": path,
                "group_dir": path.parent.name,
                "filename": path.name,
            }
        )
    return grouped


def ensure_empty_fault_surface_dir(target_dir: Path) -> None:
    if not target_dir.exists():
        return
    for path in target_dir.glob("*.vtp"):
        path.unlink()


def build_manifest_df(
    unit_id: str,
    block_x: int,
    block_y: int,
    file_records: list[dict[str, Any]],
    summary_lookup: dict[tuple[str, int, int], dict[str, Any]],
    target_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in file_records:
        summary = summary_lookup.get((str(item["fault_name"]), int(item["cell_i"]), int(item["cell_j"])), {})
        rows.append(
            {
                "UnitID": str(unit_id),
                "BlockX": int(block_x),
                "BlockY": int(block_y),
                "FaultName": str(item["fault_name"]),
                "FaultPatchFile": str(item["filename"]),
                "FaultGroupDir": str(item["group_dir"]),
                "FaultPatchSourcePath": str(item["path"]),
                "FaultPatchTargetPath": str(target_dir / str(item["filename"])),
                "CellI": int(item["cell_i"]),
                "CellJ": int(item["cell_j"]),
                "Area3D": summary.get("area_3d"),
                "CenterX": summary.get("cx"),
                "CenterY": summary.get("cy"),
                "CenterTIME": summary.get("cz"),
                "BBoxXMin": summary.get("bbox_xmin"),
                "BBoxXMax": summary.get("bbox_xmax"),
                "BBoxYMin": summary.get("bbox_ymin"),
                "BBoxYMax": summary.get("bbox_ymax"),
                "BBoxZMin": summary.get("bbox_zmin"),
                "BBoxZMax": summary.get("bbox_zmax"),
                "StrikeDeg": summary.get("strike_deg"),
                "DipDeg": summary.get("dip_deg"),
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def main() -> None:
    args = build_parser().parse_args()
    units_root = Path(args.units_root)
    summary_dir = Path(args.summary_dir) if args.summary_dir else units_root.parent / "aggregated"
    summary_dir.mkdir(parents=True, exist_ok=True)

    target_unit_ids = collect_selected_unit_ids(args, units_root=units_root)
    fault_summary_df = load_fault_summary_df(Path(args.fault_summary_csv))
    fault_summary_lookup = build_fault_summary_lookup(fault_summary_df)
    fault_file_index = index_fault_patch_files(Path(args.fault_patches_root))

    summary_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    total_copied = 0
    total_matched_files = 0

    for unit_id in target_unit_ids:
        block_x, block_y = parse_unit_id(unit_id)
        unit_dir = units_root / unit_id
        if not unit_dir.exists():
            skipped_rows.append(
                {
                    "UnitID": unit_id,
                    "BlockX": int(block_x),
                    "BlockY": int(block_y),
                    "Reason": "unit_dir_missing",
                }
            )
            continue

        target_dir = unit_dir / "fault_surfaces"
        target_dir.mkdir(parents=True, exist_ok=True)
        if bool(args.clean_target):
            ensure_empty_fault_surface_dir(target_dir)

        file_records = fault_file_index.get((block_x, block_y), [])
        copied_count = 0
        for item in file_records:
            src = Path(item["path"])
            dst = target_dir / str(item["filename"])
            if dst.exists() and not bool(args.overwrite):
                continue
            shutil.copy2(src, dst)
            copied_count += 1

        manifest_df = build_manifest_df(
            unit_id=unit_id,
            block_x=block_x,
            block_y=block_y,
            file_records=file_records,
            summary_lookup=fault_summary_lookup,
            target_dir=target_dir,
        )
        write_csv_utf8(manifest_df, unit_dir / "fault_patch_manifest.csv")
        fault_surface_summary = {
            "UnitID": str(unit_id),
            "BlockX": int(block_x),
            "BlockY": int(block_y),
            "FaultSurfaceDir": str(target_dir),
            "FaultPatchCount": int(len(file_records)),
            "CopiedFileCount": int(copied_count),
            "CleanTarget": bool(args.clean_target),
            "Overwrite": bool(args.overwrite),
        }
        write_json(unit_dir / "fault_surface_summary.json", fault_surface_summary)

        summary_rows.append(fault_surface_summary)
        total_copied += int(copied_count)
        total_matched_files += int(len(file_records))

    summary_df = pd.DataFrame(summary_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    summary_csv = summary_dir / f"{args.run_name}_unit_fault_surface_summary.csv"
    skipped_csv = summary_dir / f"{args.run_name}_skipped_units.csv"
    write_csv_utf8(summary_df, summary_csv)
    write_csv_utf8(skipped_df, skipped_csv)

    overall_summary = {
        "run_name": str(args.run_name),
        "units_root": str(units_root),
        "fault_patches_root": str(args.fault_patches_root),
        "fault_summary_csv": str(args.fault_summary_csv),
        "summary_dir": str(summary_dir),
        "selected_unit_count": int(len(target_unit_ids)),
        "processed_unit_count": int(len(summary_df)),
        "skipped_unit_count": int(len(skipped_df)),
        "matched_fault_patch_count": int(total_matched_files),
        "copied_fault_patch_count": int(total_copied),
        "summary_csv": str(summary_csv),
        "skipped_csv": str(skipped_csv),
    }
    overall_summary_path = summary_dir / f"{args.run_name}_summary.json"
    write_json(overall_summary_path, overall_summary)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title="G-DFN 单元断层面挂载",
        lines=[
            f"units_root: {units_root}",
            f"fault_patches_root: {args.fault_patches_root}",
            f"fault_summary_csv: {args.fault_summary_csv}",
            f"selected_unit_count: {overall_summary['selected_unit_count']}",
            f"processed_unit_count: {overall_summary['processed_unit_count']}",
            f"skipped_unit_count: {overall_summary['skipped_unit_count']}",
            f"matched_fault_patch_count: {overall_summary['matched_fault_patch_count']}",
            f"copied_fault_patch_count: {overall_summary['copied_fault_patch_count']}",
            f"summary_csv: {summary_csv}",
            f"summary_json: {overall_summary_path}",
        ],
    )

    print(f"summary_json: {overall_summary_path}")
    print(f"processed_unit_count: {overall_summary['processed_unit_count']}")
    print(f"matched_fault_patch_count: {overall_summary['matched_fault_patch_count']}")
    print(f"copied_fault_patch_count: {overall_summary['copied_fault_patch_count']}")


if __name__ == "__main__":
    main()
