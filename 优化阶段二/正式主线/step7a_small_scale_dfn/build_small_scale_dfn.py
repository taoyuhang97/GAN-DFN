from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
LEGACY_STEP7B_DIR = CURRENT_DIR.parent / "step7b_initial_dfn_3d"
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_step7a_small_v1.json"
if str(LEGACY_STEP7B_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_STEP7B_DIR))

import build_initial_dfn_from_3d_density_sgy as legacy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step7A small-scale/background DFN from Step6A small density.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "dfn_csv": output_dir / "small_dfn_patches.csv",
        "raw_vtk": output_dir / "small_dfn_raw_time.vtk",
        "audit_csv": output_dir / "small_dfn_generation_audit.csv",
        "summary_json": output_dir / "small_dfn_summary.json",
    }


def build_small_candidates(grid: dict[str, Any], surfaces: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates, layer_summary = legacy.build_candidate_voxels(
        density=grid["density"],
        samples=grid["samples"],
        surfaces=surfaces,
        source_trace_idx=grid["source_trace_idx"],
        config=config,
        coherence=None,
    )
    candidates["FractureScale"] = "small"
    candidates["FractureScaleCode"] = 1
    candidates["BandID"] = ""
    candidates["BandPatchOrdinal"] = 0
    candidates["BandContinuityMode"] = "small_background_density_sampling"
    candidates["BandVoxelCount"] = candidates["ComponentVoxelCount"].astype(int)
    candidates["BandLengthM"] = 0.0
    candidates["BandTimeExtentMs"] = 0.0
    candidates["BandPatchSpacingM"] = 0.0
    candidates["BandMeanDensity"] = candidates["SourceDensity"].astype(float)
    return candidates, layer_summary


def build_summary(
    config_path: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    patch_df: pd.DataFrame,
    layer_summary: dict[str, Any],
    sample_summary: dict[str, Any],
    patch_summary: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "has_patches": len(patch_df) > 0,
        "all_small_scale": bool(patch_df["FractureScale"].astype(str).eq("small").all()),
        "csv_exists": paths["dfn_csv"].exists(),
        "raw_vtk_exists": paths["raw_vtk"].exists(),
        "audit_exists": paths["audit_csv"].exists(),
        "orientation_not_all_missing": bool(patch_df["AzimuthDeg"].notna().all() and patch_df["DipDeg"].notna().all()),
        "no_oversized_small_patches": bool(
            patch_df["LengthM"].le(float(config.get("max_small_length_check_m", 110.0))).all()
            and patch_df["HeightTimeMs"].le(float(config.get("max_small_height_check_ms", 45.0))).all()
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(config_path),
        "generation_logic": "step7a_small_scale_from_step6a_small_density",
        "inputs": {
            "density_sgy": str(Path(config["density_sgy"]).resolve()),
            "trace_mapping_npz": str(Path(config["trace_mapping_npz"]).resolve()),
            "layer_dir": str(Path(config["layer_dir"]).resolve()),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected)),
        "patch_count": int(len(patch_df)),
        "layer_summary": layer_summary,
        "sample_summary": sample_summary,
        "patch_build_summary": patch_summary,
        "patch_stats": {
            "length_m": legacy.finite_stats(patch_df["LengthM"]),
            "height_time_ms": legacy.finite_stats(patch_df["HeightTimeMs"]),
            "area_m2": legacy.finite_stats(patch_df["PatchAreaM2"]),
            "source_density": legacy.finite_stats(patch_df["SourceDensity"]),
            "azimuth_deg": legacy.finite_stats(patch_df["AzimuthDeg"]),
            "dip_deg": legacy.finite_stats(patch_df["DipDeg"]),
        },
        "orientation_source_distribution": legacy.layer_distribution(patch_df["OrientationSource"]),
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)
    paths = output_paths(output_dir)
    rng = np.random.default_rng(int(config.get("random_seed", 20260715)))

    print("[step7a-small] loading small density", flush=True)
    grid = legacy.load_density_grid(Path(config["density_sgy"]).resolve(), Path(config["trace_mapping_npz"]).resolve())
    print("[step7a-small] loading surfaces", flush=True)
    surfaces = legacy.attach_surface_grids(Path(config["layer_dir"]).resolve(), grid["x_values"], grid["y_values"])
    print("[step7a-small] building candidates", flush=True)
    candidates, layer_summary = build_small_candidates(grid, surfaces, config)
    print("[step7a-small] sampling candidates", flush=True)
    selected, sample_summary = legacy.sample_candidate_voxels(candidates, config, rng)
    selected["FractureScale"] = "small"
    selected["FractureScaleCode"] = 1
    selected["BandContinuityMode"] = "small_background_density_sampling"
    print(f"[step7a-small] building patches={len(selected)}", flush=True)
    patch_df, patch_summary = legacy.build_patch_table(
        selected=selected,
        density=grid["density"],
        x_values=grid["x_values"],
        y_values=grid["y_values"],
        config=config,
        rng=rng,
    )
    patch_df["GenerationStage"] = "step7a_small_scale_density_sampling"
    patch_df["SourceType"] = "small_background_density"
    patch_df["ConstraintLevel"] = "soft"
    patch_df["Confidence"] = np.clip(pd.to_numeric(patch_df["SamplingWeight"], errors="coerce").fillna(0.0), 0.0, None)
    conf_max = float(patch_df["Confidence"].quantile(0.95)) if len(patch_df) else 1.0
    patch_df["Confidence"] = (patch_df["Confidence"] / max(conf_max, 1.0e-9)).clip(0.05, 1.0)
    patch_df["NeedsWellCorrection"] = 1

    audit_df = legacy.build_audit(patch_df)
    audit_df["ActionReason"] = "small_scale_background_from_step6a_small_density"
    patch_df.to_csv(paths["dfn_csv"], index=False, encoding="utf-8-sig")
    audit_df.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    legacy.write_legacy_vtk(
        paths["raw_vtk"],
        patch_df,
        "step7a_small_scale_dfn_raw_time",
        display=False,
        display_z_scale=float(config.get("display_z_scale", 5.0)),
        geometry_time_scale_m_per_ms=float(config.get("geometry_time_scale_m_per_ms", config.get("orientation_time_scale_m_per_ms", 1.0))),
    )
    summary = build_summary(config_path, config, paths, candidates, selected, patch_df, layer_summary, sample_summary, patch_summary)
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step7a-small] CSV: {paths['dfn_csv']}", flush=True)
    print(f"[step7a-small] VTK: {paths['raw_vtk']}", flush=True)
    print(f"[step7a-small] status={summary['status']} patch_count={len(patch_df)}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
