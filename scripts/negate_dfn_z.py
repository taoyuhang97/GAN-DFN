#!/usr/bin/env python3
"""Create an underground-coordinate DFN copy with negated Z coordinates."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


VERTEX_Z_COLUMNS = {f"V{i}Z" for i in range(1, 5)}


def convert_csv(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    changed_values = 0
    with source.open("r", encoding="utf-8-sig", newline="") as src, target.open(
        "w", encoding="utf-8-sig", newline=""
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader)
        z_indices = [index for index, name in enumerate(header) if name in VERTEX_Z_COLUMNS]
        if not z_indices:
            raise ValueError(f"CSV has no V1Z-V4Z columns: {source}")
        writer.writerow(header)
        row_count = 0
        for row in reader:
            row_count += 1
            for index in z_indices:
                if index < len(row) and row[index] not in {"", "nan", "NaN", "None"}:
                    row[index] = f"{-float(row[index]):.10g}"
                    changed_values += 1
            writer.writerow(row)
    return {
        "source": str(source),
        "target": str(target),
        "rows": row_count,
        "z_columns": [header[index] for index in z_indices],
        "negated_values": changed_values,
    }


def _point_line(values: list[str]) -> str:
    if len(values) != 3:
        raise ValueError(f"expected 3 point coordinates, got {len(values)}: {values}")
    return f"{values[0]} {values[1]} {-float(values[2]):.10g}"


def convert_vtk(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    points_negated = 0
    point_count = 0
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        header = [next(src) for _ in range(4)]
        if header[2].strip().upper() != "ASCII":
            raise ValueError(f"only legacy ASCII VTK is supported: {source}")
        dst.writelines(header)
        points_header = next(src)
        parts = points_header.split()
        if len(parts) < 3 or parts[0].upper() != "POINTS":
            raise ValueError(f"VTK POINTS block missing: {source}")
        point_count = int(parts[1])
        dst.write(points_header)

        buffer: list[str] = []
        while points_negated < point_count:
            line = next(src)
            buffer.extend(line.split())
            complete = (len(buffer) // 3) * 3
            for offset in range(0, complete, 3):
                dst.write(_point_line(buffer[offset : offset + 3]) + "\n")
                points_negated += 1
            buffer = buffer[complete:]
        if buffer:
            raise ValueError(f"unconsumed VTK point tokens for {source}: {buffer[:6]}")
        for line in src:
            dst.write(line)
    return {
        "source": str(source),
        "target": str(target),
        "point_count": point_count,
        "negated_points": points_negated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Z-negated DFN output copy.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv", action="append", default=[])
    parser.add_argument("--vtk", action="append", default=[])
    args = parser.parse_args()

    source_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_names = args.csv or [
        path.name for path in source_dir.glob("*.csv")
        if "patch" in path.name.lower() or "dfn" in path.name.lower()
    ]
    vtk_names = args.vtk or [path.name for path in source_dir.glob("*.vtk")]
    csv_set, vtk_set = set(csv_names), set(vtk_names)

    for source in source_dir.iterdir():
        if source.is_file() and source.name not in csv_set and source.name not in vtk_set:
            shutil.copy2(source, output_dir / source.name)

    csv_summaries = [convert_csv(source_dir / name, output_dir / name) for name in csv_names]
    vtk_summaries = [convert_vtk(source_dir / name, output_dir / name) for name in vtk_names]
    manifest = {
        "status": "pass",
        "coordinate_conversion": "Z_new = -Z_old",
        "preserved_attributes": "CenterTime, TIME, horizon times and other non-vertex fields remain unchanged",
        "input_dir": str(source_dir),
        "output_dir": str(output_dir),
        "csv": csv_summaries,
        "vtk": vtk_summaries,
    }
    (output_dir / "z_negative_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
