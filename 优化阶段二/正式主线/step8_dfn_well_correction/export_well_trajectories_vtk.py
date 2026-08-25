# -*- coding: utf-8 -*-
"""Export well trajectories inside a target block as a legacy ASCII VTK.

Writes the mine-area well trajectories (X, Y, TWT ms) as one polyline per well
so they can be displayed together with the DFN VTK files (same coordinate
convention: Z = TWT ms). A companion CSV carries the well names and a label
anchor point for ParaView / report annotations.

Outputs (in the Step8 output directory):
  - well_trajectories_raw_time.vtk
      CELL_DATA: WellID (int), WellName (Chinese), WellNameASCII (3D-text-safe)
  - well_trajectories_annotations.csv
  - well_trajectories_export.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# vtkVectorText (ParaView's "3D Text" source) only supports ASCII glyphs, so a
# pinyin/romanised label is generated for every well for 3D label display.
_PINYIN_MAP = {
    "车": "Che",
    "页": "Ye",
    "导": "Dao",
    "眼": "Yan",
    "斜": "Xie",
    "古": "Gu",
    "井": "Jing",
}


def ascii_label(name: str) -> str:
    return "".join(_PINYIN_MAP.get(ch, ch) for ch in name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export well trajectories as a VTK.")
    parser.add_argument("--wells-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--x-min", type=float, default=556150.0)
    parser.add_argument("--x-max", type=float, default=581650.0)
    parser.add_argument("--y-min", type=float, default=4193975.0)
    parser.add_argument("--y-max", type=float, default=4212275.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--wells", nargs="*", default=None, help="Restrict to these well names if provided.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    wells_root = args.wells_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    well_dirs = sorted(p for p in wells_root.iterdir() if p.is_dir())
    trajectories: list[dict[str, Any]] = []
    well_names = []
    for well_dir in well_dirs:
        well_name = well_dir.name
        if args.wells is not None and well_name not in args.wells:
            continue
        mains = list(well_dir.glob("*_t4_t7_real_well_main.csv"))
        if not mains:
            continue
        df = pd.read_csv(mains[0], low_memory=False)
        if not {"X", "Y", "TIME"}.issubset(df.columns):
            continue
        df = df.dropna(subset=["X", "Y", "TIME"]).copy()
        # 统一 Z 约定：TWT(ms) 一律取绝对值（正方向，与 *_raw_time.vtk DFN 一致）。
        # 原始数据若为负起算（如 -2462 ms），这里统一转成 2462 ms，避免在
        # ParaView 中与 DFN 沿 XOY 平面镜像。
        df["TIME"] = df["TIME"].abs()
        df = df[
            df["X"].between(args.x_min, args.x_max)
            & df["Y"].between(args.y_min, args.y_max)
        ].copy()
        if len(df) < 2:
            continue
        df = df.sort_values("TIME").reset_index(drop=True)
        if args.stride > 1:
            df = df.iloc[:: args.stride].reset_index(drop=True)
        points = df[["X", "Y", "TIME"]].to_numpy(dtype=np.float64)
        top = int(df["TIME"].idxmin())
        annotation = {
            "WellID": 0,
            "WellName": well_name,
            "LabelX": float(df["X"].iloc[top]),
            "LabelY": float(df["Y"].iloc[top]),
            "LabelZ": float(df["TIME"].iloc[top]),
            "TimeMinMs": float(df["TIME"].min()),
            "TimeMaxMs": float(df["TIME"].max()),
        }
        if "DEPT" in df.columns:
            annotation["DEPTMin"] = float(df["DEPT"].min())
            annotation["DEPTMax"] = float(df["DEPT"].max())
        trajectories.append({"well": well_name, "points": points, "annotation": annotation})
        well_names.append(well_name)

    if not trajectories:
        raise RuntimeError("no well trajectories found in target block")
    trajectories.sort(key=lambda item: item["well"])
    well_names = [item["well"] for item in trajectories]
    well_id = {name: i + 1 for i, name in enumerate(well_names)}
    for item in trajectories:
        item["annotation"]["WellID"] = well_id[item["well"]]

    # Build legacy ASCII POLYDATA with one polyline per well.
    lines: list[str] = []
    point_lines: list[str] = []
    cell_ids: list[int] = []
    cursor = 0
    line_count = 0
    line_size_total = 0
    for item in trajectories:
        pts = item["points"]
        point_lines.extend(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in pts)
        indices = list(range(cursor, cursor + len(pts)))
        lines.append(f"{len(pts)} {' '.join(str(i) for i in indices)}")
        cell_ids.append(well_id[item["well"]])
        line_count += 1
        line_size_total += len(pts) + 1
        cursor += len(pts)

    ascii_well_names = [ascii_label(name) for name in well_names]

    vtk_path = output_dir / "well_trajectories_raw_time.vtk"
    with open(vtk_path, "w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("well_trajectories_raw_time\n")
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {cursor} double\n")
        handle.write("\n".join(point_lines) + "\n")
        handle.write(f"LINES {line_count} {line_size_total}\n")
        handle.write("\n".join(lines) + "\n")
        handle.write(f"CELL_DATA {line_count}\n")
        handle.write("SCALARS WellID int 1\n")
        handle.write("LOOKUP_TABLE default\n")
        handle.write("\n".join(str(v) for v in cell_ids) + "\n")
        handle.write(f"FIELD FieldData 2\n")
        handle.write(f"WellName 1 {line_count} string\n")
        handle.write("\n".join(well_names) + "\n")
        handle.write(f"WellNameASCII 1 {line_count} string\n")
        handle.write("\n".join(ascii_well_names) + "\n")

    annotations = pd.DataFrame([item["annotation"] for item in trajectories])
    annotations.insert(1, "WellNameASCII", ascii_well_names)
    annotations.to_csv(output_dir / "well_trajectories_annotations.csv", index=False, encoding="utf-8")

    all_z = np.concatenate([item["points"][:, 2] for item in trajectories])
    export = {
        "status": "pass",
        "wells_root": str(wells_root),
        "target_block": {"x_min": args.x_min, "x_max": args.x_max, "y_min": args.y_min, "y_max": args.y_max},
        "well_count": int(len(trajectories)),
        "well_names": well_names,
        "point_count": int(cursor),
        "line_count": int(line_count),
        "stride": int(args.stride),
        "coordinate_contract": "X/Y metres; Z = TWT ms (same convention as *_raw_time.vtk DFN files)",
        "z_convention": "Z = |TWT| ms (positive); raw negative-start TIME is abs()-ed "
                        "so wells and DFN share the same orientation in ParaView",
        "z_range_ms": [float(all_z.min()), float(all_z.max())],
        "outputs": {
            "vtk": str(vtk_path),
            "annotations_csv": str(output_dir / "well_trajectories_annotations.csv"),
        },
    }
    (output_dir / "well_trajectories_export.json").write_text(
        json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(export, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
