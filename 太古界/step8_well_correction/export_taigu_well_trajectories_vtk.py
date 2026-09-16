#!/usr/bin/env python3
"""Export Taigu Step2 target-layer well tracks for Step8/Step9 display."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/taigu_step8_attribute_multiscale_v1.json"
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")

PINYIN_MAP = {
    "埕": "Cheng",
    "北": "Bei",
    "古": "Gu",
    "斜": "Xie",
    "桩": "Zhuang",
    "海": "Hai",
    "井": "Jing",
}


def ascii_label(name: str) -> str:
    return "".join(PINYIN_MAP.get(char, char) for char in name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Taigu well trajectories and label anchors.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--wells-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--wells", nargs="*", default=None)
    return parser.parse_args()


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read CSV: {path}") from last_error


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def load_metadata(root: Path) -> pd.DataFrame:
    path = root / "taigu_step2_well_metadata.csv"
    if not path.exists():
        return pd.DataFrame()
    table = read_csv(path, low_memory=False)
    return table.set_index("WellName", drop=False) if "WellName" in table.columns else pd.DataFrame()


def load_well_track(well_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(well_dir.glob("*.csv")):
        if path.name.startswith("taigu_step2_"):
            continue
        frame = read_csv(path, low_memory=False)
        if not {"X", "Y", "TIME"}.issubset(frame.columns):
            continue
        for column in ["X", "Y", "TIME", "MD", "TVD"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["X", "Y", "TIME"]).copy()
        frame["SourceSegmentFile"] = str(path.resolve())
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    track = pd.concat(parts, ignore_index=True, sort=False)
    dedup_columns = [column for column in ["MD", "X", "Y", "TIME"] if column in track.columns]
    track = track.drop_duplicates(subset=dedup_columns).sort_values(
        [column for column in ["MD", "TIME"] if column in track.columns]
    )
    return track.reset_index(drop=True)


def write_vtk(path: Path, trajectories: list[dict[str, Any]]) -> None:
    points: list[np.ndarray] = []
    lines: list[list[int]] = []
    cell_ids: list[int] = []
    cursor = 0
    for item in trajectories:
        xyz = item["points"]
        points.append(xyz)
        indices = list(range(cursor, cursor + len(xyz)))
        lines.append(indices)
        cell_ids.append(int(item["well_id"]))
        cursor += len(xyz)
    xyz = np.vstack(points)
    total_line_size = sum(len(line) + 1 for line in lines)
    text = [
        "# vtk DataFile Version 3.0",
        "taigu_well_trajectories_raw_time",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(xyz)} double",
    ]
    text.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in xyz)
    text.append(f"LINES {len(lines)} {total_line_size}")
    text.extend(f"{len(line)} {' '.join(map(str, line))}" for line in lines)
    text.extend([f"CELL_DATA {len(lines)}", "SCALARS WellID int 1", "LOOKUP_TABLE default"])
    text.extend(str(value) for value in cell_ids)
    text.extend(["FIELD FieldData 2", f"WellName 1 {len(trajectories)} string"])
    text.extend(item["well"] for item in trajectories)
    text.append(f"WellNameASCII 1 {len(trajectories)} string")
    text.extend(ascii_label(item["well"]) for item in trajectories)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def write_well_heads_vtk(path: Path, trajectories: list[dict[str, Any]]) -> None:
    annotations = [item["annotation"] for item in trajectories]
    text = [
        "# vtk DataFile Version 3.0",
        "taigu_well_heads_raw_time",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(annotations)} double",
    ]
    text.extend(
        f"{float(row['LabelX']):.6f} {float(row['LabelY']):.6f} {float(row['LabelZ']):.6f}"
        for row in annotations
    )
    text.append(f"VERTICES {len(annotations)} {2 * len(annotations)}")
    text.extend(f"1 {index}" for index in range(len(annotations)))
    text.extend([f"POINT_DATA {len(annotations)}", "SCALARS WellID int 1", "LOOKUP_TABLE default"])
    text.extend(str(int(row["WellID"])) for row in annotations)
    text.extend(["SCALARS TemporaryTimeDepth int 1", "LOOKUP_TABLE default"])
    text.extend(str(int(row["temporary_neighbor_time_depth"])) for row in annotations)
    text.extend(["FIELD FieldData 2", f"WellName 1 {len(annotations)} string"])
    text.extend(str(row["WellName"]) for row in annotations)
    text.append(f"WellNameASCII 1 {len(annotations)} string")
    text.extend(str(row["WellNameASCII"]) for row in annotations)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    root = (args.wells_root or Path(config["real_well_samples_root"])).resolve()
    output = (args.output_dir or Path(config["output_dir"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    block = dict(config["target_block"])
    metadata = load_metadata(root)
    requested = set(args.wells) if args.wells else None
    stride = max(int(args.stride), 1)

    print("[井轨迹 1/3] 扫描并合并太古界 Step2 分段", flush=True)
    items: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for well_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        well = well_dir.name
        if requested is not None and well not in requested:
            continue
        track = load_well_track(well_dir)
        if track.empty:
            skipped.append({"WellName": well, "Reason": "no_valid_step2_track"})
            continue
        inside = (
            track["X"].between(float(block["x_min"]), float(block["x_max"]))
            & track["Y"].between(float(block["y_min"]), float(block["y_max"]))
        )
        track = track.loc[inside].copy()
        if len(track) < 2:
            skipped.append({"WellName": well, "Reason": "fewer_than_two_points_in_target_block"})
            continue
        if stride > 1:
            keep = np.unique(np.r_[np.arange(0, len(track), stride), len(track) - 1])
            track = track.iloc[keep].copy()
        top_idx = track["TIME"].idxmin()
        meta = metadata.loc[well] if not metadata.empty and well in metadata.index else pd.Series(dtype=object)
        borrowed = str(meta.get("TimeDepthSource", "")).lower().startswith("borrowed")
        annotation = {
            "WellID": 0,
            "WellName": well,
            "WellNameASCII": ascii_label(well),
            "LabelX": float(track.loc[top_idx, "X"]),
            "LabelY": float(track.loc[top_idx, "Y"]),
            "LabelZ": float(track.loc[top_idx, "TIME"]),
            "TimeMinMs": float(track["TIME"].min()),
            "TimeMaxMs": float(track["TIME"].max()),
            "MDMin": float(track["MD"].min()) if "MD" in track.columns else np.nan,
            "MDMax": float(track["MD"].max()) if "MD" in track.columns else np.nan,
            "PointCount": int(len(track)),
            "SegmentCount": int(track["SourceSegmentFile"].nunique()),
            "WellType": str(meta.get("WellType", "")),
            "TimeDepthSource": str(meta.get("TimeDepthSource", "")),
            "BorrowedFrom": str(meta.get("BorrowedFrom", "")),
            "temporary_neighbor_time_depth": int(borrowed),
        }
        items.append({"well": well, "points": track[["X", "Y", "TIME"]].to_numpy(float), "annotation": annotation})

    if not items:
        raise RuntimeError("no Taigu Step2 well trajectories intersect the configured target block")
    items.sort(key=lambda item: item["well"])
    for well_id, item in enumerate(items, start=1):
        item["well_id"] = well_id
        item["annotation"]["WellID"] = well_id

    print(f"[井轨迹 2/3] 写出 {len(items)} 口井的 VTK 和井名锚点", flush=True)
    vtk_path = output / "well_trajectories_raw_time.vtk"
    heads_vtk_path = output / "well_heads_raw_time.vtk"
    annotations_path = output / "well_trajectories_annotations.csv"
    summary_path = output / "well_trajectories_export.json"
    write_vtk(vtk_path, items)
    write_well_heads_vtk(heads_vtk_path, items)
    annotations = pd.DataFrame([item["annotation"] for item in items])
    annotations.to_csv(annotations_path, index=False, encoding="utf-8-sig")
    all_points = np.vstack([item["points"] for item in items])
    summary = {
        "status": "pass",
        "config": str(args.config.resolve()),
        "wells_root": str(root),
        "target_block": block,
        "well_count": len(items),
        "well_names": [item["well"] for item in items],
        "point_count": int(len(all_points)),
        "polyline_count": len(items),
        "stride": stride,
        "coordinate_contract": "X/Y in metres; Z = original positive TWT TIME in ms; no abs or display scaling",
        "z_range_ms": [float(all_points[:, 2].min()), float(all_points[:, 2].max())],
        "temporary_time_depth_well_count": int(annotations["temporary_neighbor_time_depth"].sum()),
        "skipped_wells": skipped,
        "outputs": {
            "trajectories_vtk": str(vtk_path),
            "well_heads_vtk": str(heads_vtk_path),
            "annotations_csv": str(annotations_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[井轨迹 3/3] 导出完成", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
