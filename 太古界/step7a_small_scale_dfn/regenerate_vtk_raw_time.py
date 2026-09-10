"""Regenerate Step7A VTK from an existing fracture_patches.csv.

This avoids rerunning the expensive density sampling when only the VTK Z
coordinate convention needs to be corrected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_taigu_small_scale_dfn import write_legacy_vtk


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"]).resolve()
    csv_path = output_dir / "fracture_patches.csv"
    vtk_path = output_dir / "fracture_patches.vtk"
    patches = pd.read_csv(csv_path)
    vtk_scale = float(config.get("vtk_z_scale_m_per_ms", 1.0))

    points: list[np.ndarray] = []
    quads: list[np.ndarray] = []
    for _, patch in patches.iterrows():
        center = np.array([patch["X"], patch["Y"], patch["TIME"] * vtk_scale])
        strike_rad = np.radians(patch["AzimuthDeg"])
        strike = np.array([np.cos(strike_rad), np.sin(strike_rad), 0.0])
        dip_dir = np.array([-np.sin(strike_rad), np.cos(strike_rad), 0.0])
        dip_rad = np.radians(patch["DipDeg"])
        dip_vec = np.array(
            [np.sin(dip_rad) * dip_dir[0], np.sin(dip_rad) * dip_dir[1], np.cos(dip_rad)]
        )
        half_l = patch["PatchLengthM"] / 2.0
        half_h = patch["PatchHeightMs"] * vtk_scale / 2.0
        corners = [
            center + half_l * strike + half_h * dip_vec,
            center + half_l * strike - half_h * dip_vec,
            center - half_l * strike - half_h * dip_vec,
            center - half_l * strike + half_h * dip_vec,
        ]
        base = len(points)
        points.extend(corners)
        quads.append(np.array([base, base + 1, base + 2, base + 3], dtype=np.int64))

    cell_data = {
        "PatchID": np.arange(len(patches), dtype=np.int64),
        "Density": patches["Density"].to_numpy(dtype=np.float64),
        "DipDeg": patches["DipDeg"].to_numpy(dtype=np.float64),
        "AzimuthDeg": patches["AzimuthDeg"].to_numpy(dtype=np.float64),
        "PatchAreaM2": patches["PatchAreaM2"].to_numpy(dtype=np.float64),
        # 1=上部复合层，2=太古界风化壳；便于在可视化软件中按层位着色/筛选。
        "LayerCode": patches["LayerGroup"].map({"上部复合层": 1, "太古界风化壳": 2}).fillna(0).to_numpy(dtype=np.int32),
        "CenterTimeMs": patches["TIME"].to_numpy(dtype=np.float64),
        "FractureScale": np.full(len(patches), 1, dtype=np.int32),
        "WindowCode": np.full(len(patches), 1, dtype=np.int32),
    }
    write_legacy_vtk(
        vtk_path,
        np.stack(points) if points else np.empty((0, 3)),
        np.stack(quads) if quads else np.empty((0, 4), dtype=np.int64),
        cell_data,
    )
    print(f"[step7a-vtk] patches={len(patches)} vtk={vtk_path} z_scale={vtk_scale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
