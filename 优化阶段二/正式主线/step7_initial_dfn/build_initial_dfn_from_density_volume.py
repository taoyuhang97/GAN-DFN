from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_initial_dfn.json"
ALLOWED_LAYERS = ["沙三段", "沙四段"]
LAYER_CODE = {"沙三段": 3, "沙四段": 4}
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 8 initial DFN from formal candidate-A density volume.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_flexible(path: Path, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read csv: {path}") from last_error


def safe_numeric(series: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def finite_stats(values: pd.Series | np.ndarray | list[float]) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": int(numeric.count()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=0)),
    }


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "dfn_csv": output_dir / "initial_dfn_fracture_patches.csv",
        "raw_vtk": output_dir / "initial_dfn_raw_time.vtk",
        "display_vtk": output_dir / "initial_dfn_display.vtk",
        "summary_json": output_dir / "initial_dfn_summary.json",
        "audit_csv": output_dir / "initial_dfn_generation_audit.csv",
    }


def load_density_volume(path: Path) -> pd.DataFrame:
    df = read_csv_flexible(path, low_memory=False)
    required = ["TraceIdx", "X", "Y", "LayerGroup", "TimeWindowMin", "TimeWindowMax", "LayerThickness"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"density volume missing required columns: {missing}")
    density_col = "PredDensity" if "PredDensity" in df.columns else "Density"
    if density_col not in df.columns:
        raise ValueError("density volume missing PredDensity/Density column")

    out = df.copy()
    for column in ["TraceIdx", "X", "Y", "TimeWindowMin", "TimeWindowMax", "LayerThickness", density_col]:
        out[column] = safe_numeric(out[column])
    out["LayerGroup"] = out["LayerGroup"].astype(str)
    out["SourceDensity"] = out[density_col].clip(lower=0.0)
    out = out[
        out["LayerGroup"].isin(ALLOWED_LAYERS)
        & out["TraceIdx"].notna()
        & out["X"].notna()
        & out["Y"].notna()
        & out["TimeWindowMin"].notna()
        & out["TimeWindowMax"].notna()
        & out["LayerThickness"].gt(0)
        & out["SourceDensity"].notna()
    ].copy()
    out["DensityCellID"] = out["TraceIdx"].astype("int64").astype(str) + "_" + out["LayerGroup"]
    return out.reset_index(drop=True)


def load_fracture_point_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "row_count": 0, "orientation_columns_found": [], "orientation_source": "missing_fracture_points_csv"}
    df = read_csv_flexible(path, low_memory=False)
    orientation_candidates = [
        "Azimuth",
        "AzimuthDeg",
        "FractureAzimuth",
        "Strike",
        "StrikeDeg",
        "Dip",
        "DipDeg",
        "FractureDip",
    ]
    found = [column for column in orientation_candidates if column in df.columns]
    layer_col = "StrataName" if "StrataName" in df.columns else "LayerGroup" if "LayerGroup" in df.columns else None
    layer_counts: dict[str, int] = {}
    density_stats: dict[str, Any] = {}
    if layer_col is not None:
        layer_counts = {
            str(key): int(value)
            for key, value in df[layer_col].astype(str).value_counts(dropna=False).sort_index().items()
        }
    if "Density" in df.columns:
        density_stats = finite_stats(safe_numeric(df["Density"]))
    return {
        "exists": True,
        "row_count": int(len(df)),
        "layer_counts": layer_counts,
        "density_stats": density_stats,
        "orientation_columns_found": found,
        "orientation_source": "step4_columns" if found else "default_layer_template_no_step4_orientation_fields",
    }


def estimate_trace_spacing(df: pd.DataFrame) -> tuple[float, float]:
    def median_positive_diff(values: pd.Series) -> float | None:
        unique = np.sort(pd.to_numeric(values, errors="coerce").dropna().unique())
        if unique.size < 2:
            return None
        diffs = np.diff(unique)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return None
        return float(np.median(diffs))

    dx = median_positive_diff(df["X"]) or 12.5
    dy = median_positive_diff(df["Y"]) or 12.5
    return dx, dy


def choose_effective_count_scale(df: pd.DataFrame, config: dict[str, Any]) -> tuple[float, float]:
    density_mass = float(df["SourceDensity"].sum())
    if density_mass <= 0:
        raise RuntimeError("density volume has no positive density mass")
    requested = float(config.get("count_scale", 0.25))
    min_count = int(config.get("min_patch_count", 0))
    max_count = int(config.get("max_patch_count", 0))
    expected = density_mass * requested
    scale = requested
    if min_count > 0 and expected < min_count:
        scale = float(min_count) / density_mass
    if max_count > 0 and density_mass * scale > max_count:
        scale = float(max_count) / density_mass
    return scale, density_mass * scale


def sample_density_cells(df: pd.DataFrame, config: dict[str, Any], rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, Any]]:
    scale, expected_total = choose_effective_count_scale(df, config)
    max_per_cell = int(config.get("max_patches_per_cell", 3))
    expected = df["SourceDensity"].to_numpy(dtype=float) * scale
    expected = np.clip(expected, 0.0, float(max_per_cell))
    base = np.floor(expected).astype(int)
    frac = expected - base
    counts = base + (rng.random(len(df)) < frac).astype(int)
    counts = np.clip(counts, 0, max_per_cell)
    selected_idx = np.repeat(np.arange(len(df)), counts)
    if selected_idx.size == 0:
        strongest = int(np.nanargmax(df["SourceDensity"].to_numpy(dtype=float)))
        selected_idx = np.asarray([strongest], dtype=int)
        counts[strongest] = 1
    selected = df.iloc[selected_idx].reset_index(drop=True).copy()
    selected["DensityCellPatchOrdinal"] = selected.groupby("DensityCellID").cumcount() + 1
    selected["ExpectedPatchCountForCell"] = expected[selected_idx]
    selected["EffectiveCountScale"] = scale
    summary = {
        "density_mass": float(df["SourceDensity"].sum()),
        "requested_count_scale": float(config.get("count_scale", 0.25)),
        "effective_count_scale": float(scale),
        "expected_patch_count": float(expected_total),
        "actual_patch_count": int(len(selected)),
        "max_patches_per_cell": max_per_cell,
        "positive_density_cell_count": int(df["SourceDensity"].gt(0).sum()),
        "sampled_density_cell_count": int((counts > 0).sum()),
    }
    return selected, summary


def layer_param(config: dict[str, Any], key: str, layer: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, dict):
        return float(value.get(layer, default))
    return float(value)


def build_patch_table(
    density_df: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dx, dy = estimate_trace_spacing(density_df)
    sampled, sample_summary = sample_density_cells(density_df, config, rng)
    target = dict(config.get("target_block", {}))
    x_min = float(target.get("x_min", density_df["X"].min()))
    x_max = float(target.get("x_max", density_df["X"].max()))
    y_min = float(target.get("y_min", density_df["Y"].min()))
    y_max = float(target.get("y_max", density_df["Y"].max()))

    density_p95 = {
        layer: max(float(group["SourceDensity"].quantile(0.95)), 1.0e-9)
        for layer, group in density_df.groupby("LayerGroup", dropna=False)
    }

    rows: list[dict[str, Any]] = []
    for patch_idx, row in sampled.iterrows():
        layer = str(row["LayerGroup"])
        source_density = float(row["SourceDensity"])
        density_norm = min(source_density / density_p95.get(layer, max(source_density, 1.0)), 3.0)
        density_factor = float(np.sqrt(max(density_norm, 0.0)))

        base_length = layer_param(config, "base_length_m", layer, 35.0)
        base_height = layer_param(config, "base_height_time_ms", layer, 16.0)
        length = base_length * (1.0 + float(config.get("length_density_gain", 0.7)) * density_factor)
        length *= float(rng.uniform(0.85, 1.15))
        height = base_height * (1.0 + float(config.get("height_density_gain", 0.45)) * density_factor)
        height *= float(rng.uniform(0.85, 1.15))
        layer_thickness = float(row["LayerThickness"])
        height = min(height, layer_thickness * float(config.get("max_height_fraction_of_layer", 0.65)))
        height = max(height, min(float(config.get("min_height_time_ms", 4.0)), layer_thickness * 0.5))

        azimuth = layer_param(config, "base_azimuth_deg", layer, 60.0)
        azimuth += float(rng.normal(0.0, float(config.get("azimuth_jitter_deg", 15.0))))
        azimuth = azimuth % 180.0
        dip = layer_param(config, "base_dip_deg", layer, 72.0)
        dip += float(rng.normal(0.0, float(config.get("dip_jitter_deg", 5.0))))
        dip = float(np.clip(dip, 45.0, 89.0))

        theta = np.deg2rad(azimuth)
        half_dx = 0.5 * length * np.cos(theta)
        half_dy = 0.5 * length * np.sin(theta)
        margin_x = abs(float(half_dx))
        margin_y = abs(float(half_dy))
        jitter_x = float(rng.uniform(-0.35 * dx, 0.35 * dx))
        jitter_y = float(rng.uniform(-0.35 * dy, 0.35 * dy))
        center_x = float(row["X"]) + jitter_x
        center_y = float(row["Y"]) + jitter_y
        center_x = float(np.clip(center_x, x_min + margin_x, x_max - margin_x))
        center_y = float(np.clip(center_y, y_min + margin_y, y_max - margin_y))

        time_min = float(row["TimeWindowMin"])
        time_max = float(row["TimeWindowMax"])
        half_height = 0.5 * height
        if time_max - time_min <= height:
            center_time = 0.5 * (time_min + time_max)
            height = max(time_max - time_min, 0.0)
            half_height = 0.5 * height
        else:
            center_time = float(rng.uniform(time_min + half_height, time_max - half_height))

        patch_id = f"init_dfn_{patch_idx + 1:06d}"
        rows.append(
            {
                "PatchID": patch_id,
                "GenerationStage": "initial_density_volume_sampling",
                "SourceTraceIdx": int(row["TraceIdx"]),
                "DensityCellID": str(row["DensityCellID"]),
                "DensityCellPatchOrdinal": int(row["DensityCellPatchOrdinal"]),
                "LayerGroup": layer,
                "LayerCode": int(LAYER_CODE[layer]),
                "CenterX": center_x,
                "CenterY": center_y,
                "CenterTime": center_time,
                "TimeWindowMin": time_min,
                "TimeWindowMax": time_max,
                "LayerThickness": layer_thickness,
                "SourceDensity": source_density,
                "DensityQuantileP95Layer": density_p95.get(layer),
                "ExpectedPatchCountForCell": float(row["ExpectedPatchCountForCell"]),
                "EffectiveCountScale": float(row["EffectiveCountScale"]),
                "LengthM": float(length),
                "HeightTimeMs": float(height),
                "AzimuthDeg": float(azimuth),
                "DipDeg": float(dip),
                "TraceGridDX": float(dx),
                "TraceGridDY": float(dy),
                "OrientationSource": "default_layer_template_no_step4_orientation_fields",
                "SizeRule": "base_size_scaled_by_density_quantile",
                "SamplingRule": "deterministic_rng_density_mass_weighted",
                "NeedsWellCorrection": 1,
            }
        )

    patch_df = pd.DataFrame(rows)
    return patch_df, {**sample_summary, "trace_spacing_x": float(dx), "trace_spacing_y": float(dy)}


def write_legacy_vtk(path: Path, patch_df: pd.DataFrame, title: str, display: bool, display_z_scale: float) -> None:
    points: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    for _, row in patch_df.iterrows():
        theta = np.deg2rad(float(row["AzimuthDeg"]))
        half_dx = 0.5 * float(row["LengthM"]) * np.cos(theta)
        half_dy = 0.5 * float(row["LengthM"]) * np.sin(theta)
        half_h = 0.5 * float(row["HeightTimeMs"])
        z0 = float(row["CenterTime"]) - half_h
        z1 = float(row["CenterTime"]) + half_h
        if display:
            z0 = -z0 / display_z_scale
            z1 = -z1 / display_z_scale
        base = len(points)
        points.extend(
            [
                (float(row["CenterX"]) - half_dx, float(row["CenterY"]) - half_dy, z0),
                (float(row["CenterX"]) + half_dx, float(row["CenterY"]) + half_dy, z0),
                (float(row["CenterX"]) + half_dx, float(row["CenterY"]) + half_dy, z1),
                (float(row["CenterX"]) - half_dx, float(row["CenterY"]) - half_dy, z1),
            ]
        )
        polygons.append([base, base + 1, base + 2, base + 3])

    scalar_columns = [
        ("PatchIndex", np.arange(1, len(patch_df) + 1), "int"),
        ("LayerCode", patch_df["LayerCode"].to_numpy(), "int"),
        ("SourceTraceIdx", patch_df["SourceTraceIdx"].to_numpy(), "int"),
        ("SourceDensity", patch_df["SourceDensity"].to_numpy(), "float"),
        ("CenterTime", patch_df["CenterTime"].to_numpy(), "float"),
        ("LengthM", patch_df["LengthM"].to_numpy(), "float"),
        ("HeightTimeMs", patch_df["HeightTimeMs"].to_numpy(), "float"),
        ("AzimuthDeg", patch_df["AzimuthDeg"].to_numpy(), "float"),
        ("DipDeg", patch_df["DipDeg"].to_numpy(), "float"),
    ]

    total_polygon_size = sum(len(poly) + 1 for poly in polygons)
    lines = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} float",
    ]
    lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    lines.append(f"POLYGONS {len(polygons)} {total_polygon_size}")
    lines.extend(f"{len(poly)} {' '.join(str(idx) for idx in poly)}" for poly in polygons)
    lines.append(f"CELL_DATA {len(polygons)}")
    for name, values, dtype in scalar_columns:
        vtk_type = "int" if dtype == "int" else "float"
        lines.append(f"SCALARS {name} {vtk_type} 1")
        lines.append("LOOKUP_TABLE default")
        if vtk_type == "int":
            lines.extend(str(int(value)) for value in values)
        else:
            lines.extend(f"{float(value):.6f}" for value in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audit(patch_df: pd.DataFrame) -> pd.DataFrame:
    audit = patch_df[
        [
            "PatchID",
            "GenerationStage",
            "DensityCellID",
            "SourceTraceIdx",
            "LayerGroup",
            "SourceDensity",
            "ExpectedPatchCountForCell",
            "EffectiveCountScale",
            "CenterX",
            "CenterY",
            "CenterTime",
            "LengthM",
            "HeightTimeMs",
            "AzimuthDeg",
            "DipDeg",
            "OrientationSource",
            "SizeRule",
            "SamplingRule",
            "NeedsWellCorrection",
        ]
    ].copy()
    audit["Action"] = "create_initial_fracture_patch"
    audit["ActionReason"] = "sampled_from_step7_density_volume_cell"
    return audit


def layer_distribution(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def build_summary(
    config_path: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    density_df: pd.DataFrame,
    patch_df: pd.DataFrame,
    generation_summary: dict[str, Any],
    fracture_point_summary: dict[str, Any],
) -> dict[str, Any]:
    target = dict(config.get("target_block", {}))
    x_min = float(target.get("x_min", density_df["X"].min()))
    x_max = float(target.get("x_max", density_df["X"].max()))
    y_min = float(target.get("y_min", density_df["Y"].min()))
    y_max = float(target.get("y_max", density_df["Y"].max()))
    density_by_layer = density_df.groupby("LayerGroup")["SourceDensity"].sum().to_dict()
    patch_by_layer = patch_df["LayerGroup"].value_counts().to_dict()
    density_mass = float(sum(float(value) for value in density_by_layer.values()))
    patch_count = int(len(patch_df))
    macro_distribution: dict[str, dict[str, float]] = {}
    for layer in ALLOWED_LAYERS:
        density_share = float(density_by_layer.get(layer, 0.0)) / density_mass if density_mass > 0 else 0.0
        patch_share = float(patch_by_layer.get(layer, 0)) / patch_count if patch_count > 0 else 0.0
        macro_distribution[layer] = {
            "density_mass": float(density_by_layer.get(layer, 0.0)),
            "density_mass_share": density_share,
            "patch_count": int(patch_by_layer.get(layer, 0)),
            "patch_count_share": patch_share,
            "absolute_share_difference": abs(density_share - patch_share),
        }

    expected = float(generation_summary["expected_patch_count"])
    expected_error = abs(float(patch_count) - expected) / expected if expected > 0 else 1.0
    checks = {
        "has_patches": patch_count > 0,
        "layers_limited_to_sha3_sha4": set(patch_df["LayerGroup"].dropna().astype(str)).issubset(set(ALLOWED_LAYERS)),
        "centers_within_target_block": bool(
            patch_df["CenterX"].between(x_min, x_max).all()
            and patch_df["CenterY"].between(y_min, y_max).all()
        ),
        "times_within_layer_windows": bool(
            patch_df["CenterTime"].ge(patch_df["TimeWindowMin"]).all()
            and patch_df["CenterTime"].le(patch_df["TimeWindowMax"]).all()
        ),
        "patch_count_matches_density_mass": bool(expected_error <= 0.05),
        "macro_layer_distribution_matches_density": bool(
            all(payload["absolute_share_difference"] <= 0.05 for payload in macro_distribution.values())
        ),
        "traceability_fields_present": bool(
            patch_df[["PatchID", "DensityCellID", "SourceTraceIdx", "SourceDensity"]].notna().all().all()
        ),
        "vtk_outputs_exist": paths["raw_vtk"].exists() and paths["display_vtk"].exists(),
        "audit_rows_match_patch_count": paths["audit_csv"].exists() and len(patch_df) > 0,
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "status": status,
        "config_path": str(config_path),
        "density_volume_csv": str(Path(config["density_volume_csv"]).resolve()),
        "fracture_points_csv": str(Path(config["fracture_points_csv"]).resolve()),
        "initial_dfn_csv": str(paths["dfn_csv"]),
        "initial_dfn_raw_vtk": str(paths["raw_vtk"]),
        "initial_dfn_display_vtk": str(paths["display_vtk"]),
        "initial_dfn_generation_audit_csv": str(paths["audit_csv"]),
        "summary_json": str(paths["summary_json"]),
        "generation_logic": "density_mass_weighted_initial_sampling",
        "target_block": target,
        "allowed_layers": ALLOWED_LAYERS,
        "density_volume": {
            "row_count": int(len(density_df)),
            "layer_distribution": layer_distribution(density_df["LayerGroup"]),
            "density_stats": finite_stats(density_df["SourceDensity"]),
            "density_mass_by_layer": {str(key): float(value) for key, value in density_by_layer.items()},
        },
        "fracture_point_calibration": fracture_point_summary,
        "generation_summary": generation_summary,
        "initial_dfn": {
            "patch_count": patch_count,
            "layer_distribution": layer_distribution(patch_df["LayerGroup"]),
            "source_trace_count": int(patch_df["SourceTraceIdx"].nunique()),
            "center_x_stats": finite_stats(patch_df["CenterX"]),
            "center_y_stats": finite_stats(patch_df["CenterY"]),
            "center_time_stats": finite_stats(patch_df["CenterTime"]),
            "length_m_stats": finite_stats(patch_df["LengthM"]),
            "height_time_ms_stats": finite_stats(patch_df["HeightTimeMs"]),
            "source_density_stats": finite_stats(patch_df["SourceDensity"]),
        },
        "macro_distribution_check": macro_distribution,
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_dir = Path(config["output_dir"]).resolve()
    paths = output_paths(output_dir)
    ensure_dir(output_dir)

    density_volume_csv = Path(config["density_volume_csv"]).resolve()
    fracture_points_csv = Path(config["fracture_points_csv"]).resolve()
    if not density_volume_csv.exists():
        raise FileNotFoundError(f"density volume not found: {density_volume_csv}")
    if not fracture_points_csv.exists():
        raise FileNotFoundError(f"fracture points not found: {fracture_points_csv}")

    density_df = load_density_volume(density_volume_csv)
    fracture_summary = load_fracture_point_summary(fracture_points_csv)
    rng = np.random.default_rng(int(config.get("random_seed", 20260702)))
    patch_df, generation_summary = build_patch_table(density_df=density_df, config=config, rng=rng)
    audit_df = build_audit(patch_df)

    patch_df.to_csv(paths["dfn_csv"], index=False, encoding="utf-8-sig")
    audit_df.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    display_z_scale = float(config.get("display_z_scale", 5.0))
    write_legacy_vtk(paths["raw_vtk"], patch_df, "initial_dfn_raw_time", display=False, display_z_scale=display_z_scale)
    write_legacy_vtk(paths["display_vtk"], patch_df, "initial_dfn_display", display=True, display_z_scale=display_z_scale)

    summary = build_summary(
        config_path=config_path,
        config=config,
        paths=paths,
        density_df=density_df,
        patch_df=patch_df,
        generation_summary=generation_summary,
        fracture_point_summary=fracture_summary,
    )
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Initial DFN CSV: {paths['dfn_csv']}")
    print(f"Initial DFN raw VTK: {paths['raw_vtk']}")
    print(f"Initial DFN display VTK: {paths['display_vtk']}")
    print(f"Summary JSON: {paths['summary_json']}")
    print(f"Patch count: {len(patch_df)} status={summary['status']}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
