#!/usr/bin/env python3
"""Validate the four formal fracture-reference and conditioning products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_high_fidelity(root: Path, expected: int) -> dict[str, Any]:
    if not (root / "COMPLETE").is_file():
        raise ValueError(f"missing COMPLETE marker: {root}")
    summary = read_json(root / "summary.json")
    index = pd.read_csv(root / "model_index.csv", encoding="utf-8-sig")
    candidates = pd.read_csv(root / "candidate_index.csv", encoding="utf-8-sig")
    if summary.get("status") != "pass" or len(index) != expected or summary.get("model_count") != expected:
        raise ValueError(f"high-fidelity count/status failure in {root}")
    if index["GroupID"].duplicated().any() or index["TrackName"].duplicated().any():
        raise ValueError(f"duplicate model or track in {root}")
    excluded = set(summary["excluded_prior_source_wells"])
    leaked = sorted(set(index["SourceWellName"]) & excluded)
    if leaked:
        raise ValueError(f"prior sample source wells leaked into high-fidelity set: {leaked}")
    if not (index["PatchCount"] > 0).all():
        raise ValueError(f"empty high-fidelity model in {root}")
    if int((candidates["Selected"]).sum()) != expected:
        raise ValueError(f"candidate selection count mismatch in {root}")
    for row in index.itertuples(index=False):
        patches_path = sample_path(root, row.PatchesCSV)
        metadata_path = sample_path(root, row.MetadataJSON)
        fracture_vtk = pv.read(sample_path(root, row.FracturesVTK))
        trajectory_vtk = pv.read(sample_path(root, row.TrajectoryVTK))
        patches = pd.read_csv(patches_path, encoding="utf-8-sig")
        if len(patches) != int(row.PatchCount) or fracture_vtk.n_cells != len(patches):
            raise ValueError(f"patch/VTK count mismatch: {row.GroupID}")
        if fracture_vtk.n_points != 4 * fracture_vtk.n_cells or trajectory_vtk.n_points < 2:
            raise ValueError(f"invalid VTK geometry: {row.GroupID}")
        if set(patches["PatchID"].astype(str)) != set(read_json(metadata_path)["patch_ids"]):
            raise ValueError(f"PatchID metadata mismatch: {row.GroupID}")
    return {
        "root": str(root),
        "model_count": int(len(index)),
        "real_model_count": int((index["TrackKind"] == "real").sum()),
        "virtual_model_count": int((index["TrackKind"] == "virtual").sum()),
        "source_well_count": int(index["SourceWellName"].nunique()),
        "patch_count": int(index["PatchCount"].sum()),
        "scale_totals": {
            "small": int(index["SmallCount"].sum()),
            "medium": int(index["MediumCount"].sum()),
            "large": int(index["LargeCount"].sum()),
        },
    }


def validate_direct(root: Path, expected: int) -> dict[str, Any]:
    if not (root / "COMPLETE").is_file():
        raise ValueError(f"missing COMPLETE marker: {root}")
    summary = read_json(root / "summary.json")
    index = pd.read_csv(root / "sample_index.csv", encoding="utf-8-sig")
    candidates = pd.read_csv(root / "candidate_cells.csv", encoding="utf-8-sig")
    if summary.get("status") != "pass" or len(index) != expected or summary.get("sample_count") != expected:
        raise ValueError(f"direct-conditioning count/status failure in {root}")
    if index["SampleID"].duplicated().any() or index[["CellIX", "CellIY"]].duplicated().any():
        raise ValueError(f"duplicate sample or cell in {root}")
    widths = index["CellXMaxM"] - index["CellXMinM"]
    heights = index["CellYMaxM"] - index["CellYMinM"]
    if not (np.isclose(widths, 300.0).all() and np.isclose(heights, 300.0).all()):
        raise ValueError(f"cell dimensions are not 300 m x 300 m in {root}")
    if not (index["NearestWellTraceDistanceM"] > np.sqrt(2.0) * 150.0 + float(summary["no_well_safety_m"])).all():
        raise ValueError(f"no-well safety contract failed in {root}")
    if not (index["PatchCount"] > 0).all() or int(candidates["Selected"].sum()) != expected:
        raise ValueError(f"empty sample or selection count mismatch in {root}")
    patch_ids: list[str] = []
    for row in index.itertuples(index=False):
        patches_path = sample_path(root, row.PatchesCSV)
        metadata_path = sample_path(root, row.MetadataJSON)
        fracture_vtk = pv.read(sample_path(root, row.FracturesVTK))
        patches = pd.read_csv(patches_path, encoding="utf-8-sig")
        if len(patches) != int(row.PatchCount) or fracture_vtk.n_cells != len(patches):
            raise ValueError(f"patch/VTK count mismatch: {row.SampleID}")
        if fracture_vtk.n_points != 4 * fracture_vtk.n_cells:
            raise ValueError(f"invalid VTK geometry: {row.SampleID}")
        metadata = read_json(metadata_path)
        if set(patches["PatchID"].astype(str)) != set(metadata["patch_ids"]):
            raise ValueError(f"PatchID metadata mismatch: {row.SampleID}")
        patch_ids.extend(patches["PatchID"].astype(str).tolist())
    if len(patch_ids) != len(set(patch_ids)):
        raise ValueError(f"patch IDs are reused across direct-conditioning samples in {root}")
    return {
        "root": str(root),
        "sample_count": int(len(index)),
        "patch_count": int(index["PatchCount"].sum()),
        "unique_patch_count": int(len(set(patch_ids))),
        "min_nearest_well_distance_m": float(index["NearestWellTraceDistanceM"].min()),
        "scale_totals": {
            "small": int(index["SmallCount"].sum()),
            "medium": int(index["MediumCount"].sum()),
            "large": int(index["LargeCount"].sum()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sand-high-fidelity",
        type=Path,
        default=Path("优化阶段二/正式主线/output/high_fidelity_reference_models_sand_v1"),
    )
    parser.add_argument(
        "--taigu-high-fidelity",
        type=Path,
        default=Path("太古界/output/high_fidelity_reference_models_taigu_v1"),
    )
    parser.add_argument(
        "--sand-direct",
        type=Path,
        default=Path("优化阶段二/正式主线/output/direct_conditioning_samples_sand_v1"),
    )
    parser.add_argument(
        "--taigu-direct",
        type=Path,
        default=Path("太古界/output/direct_conditioning_samples_taigu_v1"),
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = {
        "status": "pass",
        "sand_high_fidelity": validate_high_fidelity(args.sand_high_fidelity.resolve(), 200),
        "taigu_high_fidelity": validate_high_fidelity(args.taigu_high_fidelity.resolve(), 40),
        "sand_direct": validate_direct(args.sand_direct.resolve(), 80),
        "taigu_direct": validate_direct(args.taigu_direct.resolve(), 40),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
