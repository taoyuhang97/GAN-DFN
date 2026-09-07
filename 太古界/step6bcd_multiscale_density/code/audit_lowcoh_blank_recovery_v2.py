from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit medium/large DFN coverage in the low-coherence target window.")
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--version", default="formal_demo_10km_multiscale_flow_v2")
    parser.add_argument("--x-min", type=float, default=578770.0)
    parser.add_argument("--x-max", type=float, default=578870.0)
    parser.add_argument("--y-min", type=float, default=4200900.0)
    parser.add_argument("--y-max", type=float, default=4201100.0)
    parser.add_argument("--time-min", type=float, default=2797.0)
    parser.add_argument("--time-max", type=float, default=3005.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def overlap_count(frame: pd.DataFrame, bounds: dict[str, float]) -> int:
    if frame.empty:
        return 0
    column_sets = [
        ("x_min", "x_max", "y_min", "y_max", "time_min_ms", "time_max_ms"),
        ("XMin", "XMax", "YMin", "YMax", "TimeMin", "TimeMax"),
    ]
    for columns in column_sets:
        if all(column in frame.columns for column in columns):
            x0, x1, y0, y1, t0, t1 = (pd.to_numeric(frame[column], errors="coerce") for column in columns)
            mask = (
                (x1 >= bounds["x_min"])
                & (x0 <= bounds["x_max"])
                & (y1 >= bounds["y_min"])
                & (y0 <= bounds["y_max"])
                & (t1 >= bounds["time_min"])
                & (t0 <= bounds["time_max"])
            )
            return int(mask.fillna(False).sum())
    vertex_x = [column for column in frame.columns if re.fullmatch(r"V[1-4]X", column)]
    vertex_y = [column for column in frame.columns if re.fullmatch(r"V[1-4]Y", column)]
    vertex_z = [column for column in frame.columns if re.fullmatch(r"V[1-4]Z", column)]
    if vertex_x and len(vertex_x) == len(vertex_y) == len(vertex_z):
        x = frame[vertex_x].apply(pd.to_numeric, errors="coerce")
        y = frame[vertex_y].apply(pd.to_numeric, errors="coerce")
        t = frame[vertex_z].apply(pd.to_numeric, errors="coerce")
        mask = (
            (x.max(axis=1) >= bounds["x_min"])
            & (x.min(axis=1) <= bounds["x_max"])
            & (y.max(axis=1) >= bounds["y_min"])
            & (y.min(axis=1) <= bounds["y_max"])
            & (t.max(axis=1) >= bounds["time_min"])
            & (t.min(axis=1) <= bounds["time_max"])
        )
        return int(mask.fillna(False).sum())
    if {"CenterX", "CenterY", "CenterTime"}.issubset(frame.columns):
        x = pd.to_numeric(frame["CenterX"], errors="coerce")
        y = pd.to_numeric(frame["CenterY"], errors="coerce")
        t = pd.to_numeric(frame["CenterTime"], errors="coerce")
        mask = (
            x.between(bounds["x_min"], bounds["x_max"])
            & y.between(bounds["y_min"], bounds["y_max"])
            & t.between(bounds["time_min"], bounds["time_max"])
        )
        return int(mask.fillna(False).sum())
    return 0


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def main() -> int:
    args = parse_args()
    root = args.formal_root.resolve()
    version = args.version
    bounds = {
        "x_min": args.x_min,
        "x_max": args.x_max,
        "y_min": args.y_min,
        "y_max": args.y_max,
        "time_min": args.time_min,
        "time_max": args.time_max,
    }
    paths = {
        "step6b": root / f"step6b_demo_density_volume_3d/output/{version}/step6b_medium/medium_corridor_component_summary.csv",
        "step6c": root / f"step6b_demo_density_volume_3d/output/{version}/step6c_large/large_fault_component_summary.csv",
        "step7b": root / f"step7b_multiscale_initial_dfn/output/{version}/medium_dfn_patches.csv",
        "step7c": root / f"step7c_large_fault_dfn/output/{version}/large_inferred_fault_surface_patches.csv",
        "step8": root / f"step8_dfn_well_correction/output/{version}/well_corrected_dfn_fracture_patches.csv",
    }
    frames = {name: read_csv(path) for name, path in paths.items()}
    result = {
        "status": "pass",
        "window": bounds,
        "paths": {name: str(path) for name, path in paths.items()},
        "total_rows": {name: int(len(frame)) for name, frame in frames.items()},
        "window_overlap_rows": {name: overlap_count(frame, bounds) for name, frame in frames.items()},
        "branch_counts": {
            name: (
                {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()}
                if (column := next((candidate for candidate in ("candidate_branch", "CandidateBranch") if candidate in frame.columns), None))
                else {}
            )
            for name, frame in frames.items()
        },
    }
    step9_dir = root / f"step9_section_visualize/output/{version}/cheye1_dfn_multibackground_sections"
    result["step9"] = {
        "output_dir": str(step9_dir),
        "png_count": int(len(list(step9_dir.rglob("*.png")))) if step9_dir.exists() else 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
