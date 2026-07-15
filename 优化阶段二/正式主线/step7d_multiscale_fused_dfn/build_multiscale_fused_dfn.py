from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_step7d_fused_v1.json"
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse Step7A/B/C multiscale DFN outputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read csv: {path}") from last_error


def finite_stats(values: Any) -> dict[str, float | int | None]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if arr.empty:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": int(arr.count()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(arr.median()),
        "std": float(arr.std(ddof=0)),
    }


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "dfn_csv": output_dir / "fused_multiscale_dfn_patches.csv",
        "raw_vtk": output_dir / "fused_multiscale_dfn_raw_time.vtk",
        "summary_json": output_dir / "fused_multiscale_summary.json",
        "audit_csv": output_dir / "fused_multiscale_audit.csv",
    }


def normalize_input(df: pd.DataFrame, scale: str, source_file: Path) -> pd.DataFrame:
    out = df.copy()
    required = ["PatchID", "CenterX", "CenterY", "CenterTime", "LengthM", "HeightTimeMs", "AzimuthDeg", "DipDeg"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"{source_file} missing columns: {missing}")
    out["FractureScale"] = scale
    out["FractureScaleCode"] = {"small": 1, "medium": 2, "large": 3}[scale]
    if "SourceType" not in out.columns:
        out["SourceType"] = f"{scale}_unknown"
    if "ConstraintLevel" not in out.columns:
        out["ConstraintLevel"] = "soft" if scale == "small" else "seismic_prior"
    if "Confidence" not in out.columns:
        out["Confidence"] = 0.55 if scale == "small" else 0.75 if scale == "medium" else 0.90
    for col in ["CenterX", "CenterY", "CenterTime", "LengthM", "HeightTimeMs", "AzimuthDeg", "DipDeg", "PatchAreaM2", "SourceDensity", "Confidence"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "PatchAreaM2" not in out.columns:
        out["PatchAreaM2"] = out["LengthM"] * out["HeightTimeMs"]
    out["Step7SourceFile"] = str(source_file)
    return out


def downsample_small_near_major(small: pd.DataFrame, major: pd.DataFrame, config: dict[str, Any], rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, Any]]:
    if small.empty or major.empty:
        return small, {"removed_count": 0, "reason": "empty_small_or_major"}
    radius = float(config.get("small_downsample_near_major_radius_m", 80.0))
    keep_fraction = float(config.get("small_keep_fraction_near_major", 0.45))
    major_xy = major[["CenterX", "CenterY"]].to_numpy(dtype=float)
    tree = cKDTree(major_xy)
    small_xy = small[["CenterX", "CenterY"]].to_numpy(dtype=float)
    dist, _idx = tree.query(small_xy, k=1)
    near = dist <= radius
    keep = np.ones(len(small), dtype=bool)
    near_idx = np.where(near)[0]
    if near_idx.size:
        keep_random = rng.random(near_idx.size) < keep_fraction
        keep[near_idx] = keep_random
    out = small.iloc[keep].reset_index(drop=True).copy()
    return out, {
        "radius_m": radius,
        "keep_fraction_near_major": keep_fraction,
        "near_major_count": int(near.sum()),
        "removed_count": int(len(small) - len(out)),
        "input_small_count": int(len(small)),
        "output_small_count": int(len(out)),
    }


def add_render_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    density = pd.to_numeric(out.get("SourceDensity", pd.Series(np.zeros(len(out)))), errors="coerce").fillna(0.0)
    geological = ~out["FractureScale"].astype(str).eq("large")
    valid = density[geological & density.notna()]
    clip_max = float(valid.quantile(0.99)) if not valid.empty else float(density.quantile(0.99) if len(density) else 1.0)
    clip_max = max(clip_max, 1.0e-6)
    out["SourceDensityRender"] = density.clip(0.0, clip_max)
    out["SourceDensityRenderNorm"] = (out["SourceDensityRender"] / clip_max).clip(0.0, 1.0)
    out["SourceDensityRenderClipMax"] = clip_max
    return out


def patch_vertices(row: pd.Series) -> list[tuple[float, float, float]]:
    azimuth = np.deg2rad(float(row["AzimuthDeg"]))
    dip = np.deg2rad(float(np.clip(row["DipDeg"], 1.0, 89.9)))
    half_length = 0.5 * float(row["LengthM"])
    half_height_time = 0.5 * float(row["HeightTimeMs"])
    strike = np.asarray([np.cos(azimuth), np.sin(azimuth)], dtype=float)
    dip_horizontal = np.asarray([-np.sin(azimuth), np.cos(azimuth)], dtype=float)
    horizontal_dip_half = half_height_time / max(np.tan(dip), 1.0e-6)
    center_xy = np.asarray([float(row["CenterX"]), float(row["CenterY"])], dtype=float)
    center_t = float(row["CenterTime"])
    corners = []
    for strike_sign, dip_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        xy = center_xy + strike_sign * half_length * strike + dip_sign * horizontal_dip_half * dip_horizontal
        corners.append((float(xy[0]), float(xy[1]), float(center_t + dip_sign * half_height_time)))
    return corners


def write_vtk(path: Path, df: pd.DataFrame, title: str) -> None:
    points: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    for _, row in df.iterrows():
        base = len(points)
        points.extend(patch_vertices(row))
        polygons.append([base, base + 1, base + 2, base + 3])
    lines = ["# vtk DataFile Version 3.0", title, "ASCII", "DATASET POLYDATA", f"POINTS {len(points)} float"]
    lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    lines.append(f"POLYGONS {len(polygons)} {sum(len(poly)+1 for poly in polygons)}")
    lines.extend(f"{len(poly)} {' '.join(str(idx) for idx in poly)}" for poly in polygons)
    lines.append(f"CELL_DATA {len(polygons)}")
    scalar_cols = [
        ("FractureScaleCode", "int"),
        ("SourceDensity", "float"),
        ("SourceDensityRenderNorm", "float"),
        ("Confidence", "float"),
        ("LengthM", "float"),
        ("HeightTimeMs", "float"),
        ("PatchAreaM2", "float"),
        ("AzimuthDeg", "float"),
        ("DipDeg", "float"),
    ]
    for col, typ in scalar_cols:
        lines.append(f"SCALARS {col} {typ} 1")
        lines.append("LOOKUP_TABLE default")
        values = np.nan_to_num(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if typ == "int":
            lines.extend(str(int(v)) for v in values)
        else:
            lines.extend(f"{float(v):.6f}" for v in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)
    paths = output_paths(output_dir)
    rng = np.random.default_rng(int(config.get("random_seed", 20260715)))

    small = normalize_input(read_csv_flexible(Path(config["small_dfn_csv"]).resolve()), "small", Path(config["small_dfn_csv"]).resolve())
    medium = normalize_input(read_csv_flexible(Path(config["medium_dfn_csv"]).resolve()), "medium", Path(config["medium_dfn_csv"]).resolve())
    large = normalize_input(read_csv_flexible(Path(config["large_dfn_csv"]).resolve()), "large", Path(config["large_dfn_csv"]).resolve())
    small_filtered, small_filter_summary = downsample_small_near_major(small, pd.concat([medium, large], ignore_index=True), config, rng)
    fused = pd.concat([large, medium, small_filtered], ignore_index=True)
    fused["PatchID"] = [f"fused_multiscale_{idx + 1:06d}" for idx in range(len(fused))]
    fused = add_render_columns(fused)
    fused.to_csv(paths["dfn_csv"], index=False, encoding="utf-8-sig")
    audit = fused[["PatchID", "FractureScale", "SourceType", "ConstraintLevel", "CenterX", "CenterY", "CenterTime", "LengthM", "HeightTimeMs", "AzimuthDeg", "DipDeg", "PatchAreaM2", "Step7SourceFile"]].copy()
    audit["Action"] = "keep_multiscale_patch"
    audit.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    write_vtk(paths["raw_vtk"], fused, "step7d_fused_multiscale_dfn_raw_time")
    summary = {
        "status": "pass",
        "config_path": str(config_path),
        "generation_logic": "step7d_large_hard_medium_prior_small_background_fusion",
        "inputs": {
            "small_dfn_csv": str(Path(config["small_dfn_csv"]).resolve()),
            "medium_dfn_csv": str(Path(config["medium_dfn_csv"]).resolve()),
            "large_dfn_csv": str(Path(config["large_dfn_csv"]).resolve()),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "input_counts": {"small": int(len(small)), "medium": int(len(medium)), "large": int(len(large))},
        "small_filter_summary": small_filter_summary,
        "patch_count": int(len(fused)),
        "scale_counts": {str(k): int(v) for k, v in fused["FractureScale"].value_counts(dropna=False).items()},
        "source_type_counts": {str(k): int(v) for k, v in fused["SourceType"].value_counts(dropna=False).items()},
        "patch_stats": {
            "length_m": finite_stats(fused["LengthM"]),
            "height_time_ms": finite_stats(fused["HeightTimeMs"]),
            "area_m2": finite_stats(fused["PatchAreaM2"]),
            "azimuth_deg": finite_stats(fused["AzimuthDeg"]),
            "dip_deg": finite_stats(fused["DipDeg"]),
        },
        "checks": {
            "has_all_scales": set(fused["FractureScale"].astype(str)).issuperset({"small", "medium", "large"}),
            "csv_exists": paths["dfn_csv"].exists(),
            "raw_vtk_exists": paths["raw_vtk"].exists(),
            "audit_exists": paths["audit_csv"].exists(),
        },
    }
    summary["status"] = "pass" if all(bool(v) for v in summary["checks"].values()) else "fail"
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step7d-fused] CSV: {paths['dfn_csv']}", flush=True)
    print(f"[step7d-fused] VTK: {paths['raw_vtk']}", flush=True)
    print(f"[step7d-fused] status={summary['status']} patch_count={len(fused)}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
