from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_unified_t4_t7_density_samples.json"
ALLOWED_LAYERS = ["沙三段", "沙四段"]
ATTRIBUTE_COLUMNS = ["SeisAmp", "Coherence", "AntTrack", "CurvatureMax"]
OUTPUT_COLUMNS = [
    "SourceKind", "SourceWellName", "TrackWellName",
    "X", "Y", "TIME", "LayerGroup",
    "PresenceLabel", "DensityLabel", "HasFracture", "PointConfidence", "SampleWeight",
    *ATTRIBUTE_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curvature-led Step5B unified T4-T7 samples.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--virtual-samples-csv", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_step4_inputs(manifest_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifest = read_json(manifest_path)
    prediction_rel = manifest.get("step5_input") or manifest.get("prediction_table")
    points_rel = manifest.get("fracture_point_table")
    if not prediction_rel or not points_rel:
        raise RuntimeError("Step4 manifest lacks the merged prediction or fracture-point table")
    prediction_csv = (manifest_path.parent / str(prediction_rel)).resolve()
    points_csv = (manifest_path.parent / str(points_rel)).resolve()
    if not prediction_csv.exists() or not points_csv.exists():
        raise FileNotFoundError("Step4 manifest references a missing input file")
    return prediction_csv, points_csv, manifest


def numeric(series: pd.Series | Any, index: pd.Index | None = None) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(pd.Series(series, index=index), errors="coerce")


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def add_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out[OUTPUT_COLUMNS]


def load_real_attributes(roots: list[Path], target_wells: set[str]) -> pd.DataFrame:
    keep = ["WellName", "DEPT", *ATTRIBUTE_COLUMNS]
    frames: list[pd.DataFrame] = []
    seen_paths: set[Path] = set()
    for root in roots:
        for path in sorted(root.glob("*/*_t4_t7_real_well_main.csv")):
            path = path.resolve()
            if path in seen_paths or path.parent.name not in target_wells:
                continue
            seen_paths.add(path)
            header = pd.read_csv(path, nrows=0).columns
            usecols = [column for column in keep if column in header]
            frame = pd.read_csv(path, usecols=usecols)
            frames.append(frame)
    if not frames:
        raise RuntimeError("No Step2 real-well attribute tables matched the Step4 wells")
    out = pd.concat(frames, ignore_index=True)
    require_columns(out, ["WellName", "DEPT", *ATTRIBUTE_COLUMNS], "Step2 real attributes")
    if out.duplicated(["WellName", "DEPT"]).any():
        raise RuntimeError("Step2 attribute roots contain duplicate WellName+DEPT rows")
    return out


def normalize_real(real_df: pd.DataFrame, attr_df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "SampleID", "WellName", "X", "Y", "TIME", "DEPT", "StrataName",
        "Density", "HasFracture", "PredictionValid",
    ]
    require_columns(real_df, required, "Step4 merged prediction")
    if real_df.duplicated(["WellName", "DEPT"]).any():
        raise RuntimeError("Step4 input is not the unique merged WellName+DEPT table")
    out = real_df.merge(attr_df.drop(columns=["SampleID"], errors="ignore"), on=["WellName", "DEPT"], how="left", validate="one_to_one")
    out = out[out["PredictionValid"].eq(1) & out["StrataName"].isin(ALLOWED_LAYERS)].copy()
    if numeric(out["CurvatureMax"]).isna().any():
        missing = int(numeric(out["CurvatureMax"]).isna().sum())
        raise RuntimeError(f"{missing} real rows lack mandatory aligned CurvatureMax")
    presence = numeric(out["HasFracture"]).gt(0).astype(int)
    out["PresenceLabel"] = presence
    out["DensityLabel"] = numeric(out["Density"]).where(presence.eq(1))
    out["SourceKind"] = "real_well"
    out["SourceWellName"] = out["WellName"].astype(str)
    out["TrackWellName"] = out["WellName"].astype(str)
    out["LayerGroup"] = out["StrataName"]
    out["SampleWeight"] = 1.0
    out["PointConfidence"] = 1.0
    return add_output_columns(out)


def normalize_virtual(df: pd.DataFrame, real_sample_ids: set[str]) -> pd.DataFrame:
    required = [
        "SourceSampleID", "SourceWellName", "VirtualWellName", "X", "Y", "TIME",
        "StrataName", "PresenceLabel", "DensityLabel", "PointConfidence", "SampleWeight",
        *ATTRIBUTE_COLUMNS,
    ]
    require_columns(df, required, "Step5A virtual training samples")
    out = df.copy()
    if "CurvaturePos" in out.columns:
        raise RuntimeError("CurvaturePos is forbidden in the formal Step5 contract")
    out = out[numeric(out["PresenceLabel"]).notna() & numeric(out["CurvatureMax"]).notna()].copy()
    if not set(out["SourceSampleID"].astype(str)).issubset(real_sample_ids):
        raise RuntimeError("Virtual SourceSampleID contains rows absent from the current Step4 merged table")
    presence = numeric(out["PresenceLabel"]).gt(0).astype(int)
    density_label = numeric(out["DensityLabel"])
    if density_label[presence.eq(1)].isna().any():
        raise RuntimeError("Positive virtual presence labels require conditional density labels")
    if density_label[presence.eq(0)].notna().any():
        raise RuntimeError("Negative virtual presence labels must keep conditional density unknown")
    out["PresenceLabel"] = presence
    out["HasFracture"] = presence
    out["DensityLabel"] = density_label
    out["SourceKind"] = "virtual_well"
    out["TrackWellName"] = out["VirtualWellName"].astype(str)
    out["LayerGroup"] = out["StrataName"].where(out["StrataName"].isin(ALLOWED_LAYERS))
    out["PointConfidence"] = numeric(out["PointConfidence"]).clip(0, 1)
    out["SampleWeight"] = numeric(out["SampleWeight"]).clip(lower=0)
    out = out[out["LayerGroup"].notna()].copy()
    return add_output_columns(out)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    manifest_path = Path(config["step4_manifest"]).resolve()
    real_csv, _points_csv, manifest = resolve_step4_inputs(manifest_path)
    virtual_csv = args.virtual_samples_csv or Path(config["virtual_well_training_samples_csv"])
    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    real_df = pd.read_csv(real_csv)
    target_wells = set(real_df["WellName"].dropna().astype(str))
    attr_df = load_real_attributes([Path(path) for path in config["real_well_attribute_roots"]], target_wells)
    real_ids = set(real_df["SampleID"].dropna().astype(str))
    real_out = normalize_real(real_df, attr_df)

    output_csv = output_dir / "unified_t4_t7_density_samples.csv"
    summary_json = output_dir / "unified_t4_t7_density_samples_summary.json"
    qc_csv = output_dir / "unified_t4_t7_qc.csv"
    contract_csv = output_dir / "unified_t4_t7_field_contract.csv"
    output_tmp = output_csv.with_name(f"{output_csv.name}.tmp")
    real_out.to_csv(output_tmp, index=False, encoding="utf-8-sig")

    chunksize = int(config.get("chunksize", 250000))
    virtual_rows = 0
    virtual_presence_positive = 0
    virtual_density_rows = 0
    virtual_weight = 0.0
    virtual_source_wells: set[str] = set()
    virtual_tracks: set[str] = set()
    virtual_layer_counts: dict[str, int] = {}
    virtual_checks = {
        "presence_labels_complete": True,
        "positive_density_labels_complete": True,
        "negative_density_labels_null": True,
        "curvature_max_complete": True,
        "virtual_rows_training_ready": True,
    }
    for chunk_idx, chunk in enumerate(pd.read_csv(virtual_csv, chunksize=chunksize), start=1):
        normalized = normalize_virtual(chunk, real_ids)
        if not normalized.empty:
            normalized.to_csv(output_tmp, mode="a", header=False, index=False, encoding="utf-8")
            presence = numeric(normalized["PresenceLabel"])
            density = numeric(normalized["DensityLabel"])
            virtual_rows += int(len(normalized))
            virtual_presence_positive += int(presence.sum())
            virtual_density_rows += int(density.notna().sum())
            virtual_weight += float(numeric(normalized["SampleWeight"]).sum())
            virtual_source_wells.update(normalized["SourceWellName"].dropna().astype(str).unique())
            virtual_tracks.update(normalized["TrackWellName"].dropna().astype(str).unique())
            for layer, count in normalized["LayerGroup"].value_counts().items():
                virtual_layer_counts[str(layer)] = virtual_layer_counts.get(str(layer), 0) + int(count)
            virtual_checks["presence_labels_complete"] &= bool(presence.notna().all())
            virtual_checks["positive_density_labels_complete"] &= bool(density[presence.eq(1)].notna().all())
            virtual_checks["negative_density_labels_null"] &= bool(density[presence.eq(0)].isna().all())
            virtual_checks["curvature_max_complete"] &= bool(numeric(normalized["CurvatureMax"]).notna().all())
            virtual_checks["virtual_rows_training_ready"] &= bool(
                presence.notna().all() & numeric(normalized["PointConfidence"]).notna().all()
            )
        print(f"[step5b] virtual_chunk={chunk_idx} accepted_rows={virtual_rows}", flush=True)

    real_weight = float(numeric(real_out["SampleWeight"]).sum())
    max_ratio = float(config.get("max_virtual_to_real_weight_ratio", 1.0))
    real_presence = numeric(real_out["PresenceLabel"])
    real_density = numeric(real_out["DensityLabel"])
    checks = {
        "uses_current_step4_manifest": manifest.get("step5_input") == manifest.get("prediction_table"),
        "real_keys_unique": not real_df.duplicated(["WellName", "DEPT"]).any(),
        "has_real_and_virtual": not real_out.empty and virtual_rows > 0,
        "presence_labels_complete": bool(real_presence.notna().all()) and virtual_checks["presence_labels_complete"],
        "positive_density_labels_complete": bool(real_density[real_presence.eq(1)].notna().all()) and virtual_checks["positive_density_labels_complete"],
        "negative_density_labels_null": bool(real_density[real_presence.eq(0)].isna().all()) and virtual_checks["negative_density_labels_null"],
        "curvature_max_complete": bool(numeric(real_out["CurvatureMax"]).notna().all()) and virtual_checks["curvature_max_complete"],
        "virtual_rows_training_ready": virtual_rows > 0 and virtual_checks["virtual_rows_training_ready"],
        "virtual_weight_bounded": virtual_weight <= real_weight * max_ratio + 1.0e-9,
        "curvature_pos_excluded": "CurvaturePos" not in OUTPUT_COLUMNS,
    }
    status = "pass" if all(checks.values()) else "fail"
    real_layer_counts = {str(k): int(v) for k, v in real_out["LayerGroup"].value_counts().items()}
    layer_counts = dict(real_layer_counts)
    for layer, count in virtual_layer_counts.items():
        layer_counts[layer] = layer_counts.get(layer, 0) + count
    source_kind_summary = [
        {"SourceKind": "real_well", "rows": int(len(real_out)), "presence_positive": int(real_presence.sum()),
         "density_label_rows": int(real_density.notna().sum()), "sample_weight": real_weight,
         "source_wells": int(real_out["SourceWellName"].nunique()), "tracks": int(real_out["TrackWellName"].nunique())},
        {"SourceKind": "virtual_well", "rows": int(virtual_rows), "presence_positive": int(virtual_presence_positive),
         "density_label_rows": int(virtual_density_rows), "sample_weight": virtual_weight,
         "source_wells": int(len(virtual_source_wells)), "tracks": int(len(virtual_tracks))},
    ]
    pd.DataFrame([
        {"SourceKind": row["SourceKind"], "Rows": row["rows"], "WeightSum": row["sample_weight"],
         "PresencePositiveRows": row["presence_positive"], "DensityLabelRows": row["density_label_rows"]}
        for row in source_kind_summary
    ]).to_csv(qc_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame({"Field": OUTPUT_COLUMNS, "RequiredInFormalOutput": True}).to_csv(contract_csv, index=False, encoding="utf-8-sig")
    payload = {
        "status": status,
        "step4_manifest": str(manifest_path),
        "step4_manifest_version": manifest.get("version"),
        "real_well_prediction_csv": str(real_csv),
        "virtual_well_training_samples_csv": str(virtual_csv),
        "output_csv": str(output_csv),
        "summary": {
            "total_rows": int(len(real_out) + virtual_rows),
            "source_kind": source_kind_summary,
            "layer_counts": layer_counts,
            "presence_positive_rows": int(real_presence.sum() + virtual_presence_positive),
            "density_label_rows": int(real_density.notna().sum() + virtual_density_rows),
        },
        "real_weight_sum": real_weight,
        "virtual_weight_sum": virtual_weight,
        "virtual_to_real_weight_ratio": virtual_weight / real_weight if real_weight else None,
        "checks": {key: bool(value) for key, value in checks.items()},
    }
    write_json(summary_json, payload)
    if status == "pass":
        output_tmp.replace(output_csv)
    else:
        output_tmp.unlink(missing_ok=True)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
