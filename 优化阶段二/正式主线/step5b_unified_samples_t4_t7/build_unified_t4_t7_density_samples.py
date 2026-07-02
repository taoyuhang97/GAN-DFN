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
ATTRIBUTE_COLUMNS = ["SeisAmp", "Coherence", "AntTrack", "CurvatureMax", "CurvaturePos"]
STAT_SUFFIXES = ["Mean", "Std", "Min", "Max", "ValidCount"]
REQUIRED_OUTPUT_COLUMNS = [
    "SampleID",
    "SourceKind",
    "SourceWellName",
    "TrackWellName",
    "WellName",
    "X",
    "Y",
    "TIME",
    "TVD",
    "DEPT",
    "StrataName",
    "LayerGroup",
    "Density",
    "DensityLabel",
    "HasFracture",
    "PointConfidence",
    *ATTRIBUTE_COLUMNS,
    *[f"{attr}{suffix}" for attr in ATTRIBUTE_COLUMNS for suffix in STAT_SUFFIXES],
]
OPTIONAL_OUTPUT_COLUMNS = [
    "SourceSampleID",
    "VirtualWellName",
    "DistanceToSource",
    "AttributeContinuity",
    "ValidContinuityAttributeCount",
    "SourceDensity",
    "DensitySourceLogic",
    "DistanceConfidenceWeight",
    "ConfidenceLogic",
    "SingleSourceCheck",
    "IsRefinedFracturePoint",
]
OUTPUT_COLUMNS = [*REQUIRED_OUTPUT_COLUMNS, *OPTIONAL_OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 5B formal unified T4-T7 density samples.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_numeric(series: pd.Series | Any, index: pd.Index | None = None) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(pd.Series(series, index=index), errors="coerce")


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def layer_group_from_strata(series: pd.Series) -> pd.Series:
    return series.astype(str).where(series.astype(str).isin(ALLOWED_LAYERS), pd.NA)


def normalize_has_fracture(df: pd.DataFrame) -> pd.Series:
    return safe_numeric(df.get("HasFracture", pd.Series(0, index=df.index))).fillna(0).gt(0).astype(int)


def add_missing_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out[OUTPUT_COLUMNS]


def normalize_density_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    density = safe_numeric(out.get("Density", pd.Series(np.nan, index=out.index)))
    out["Density"] = density.clip(lower=0)
    out["DensityLabel"] = out["Density"].fillna(0.0)
    out["HasFracture"] = normalize_has_fracture(out)
    return out


def normalize_real_samples(real_df: pd.DataFrame, refined_point_ids: set[str]) -> pd.DataFrame:
    require_columns(real_df, ["SampleID", "WellName", "X", "Y", "TIME", "Density", "HasFracture", "StrataName"], "real well prediction")
    out = real_df.copy()
    out["SampleID"] = out["SampleID"].astype(str)
    out["SourceKind"] = "real_well"
    out["SourceWellName"] = out["WellName"].astype(str)
    out["TrackWellName"] = out["WellName"].astype(str)
    out["LayerGroup"] = layer_group_from_strata(out["StrataName"])
    out["PointConfidence"] = 1.0
    out["SourceSampleID"] = out["SampleID"]
    out["VirtualWellName"] = pd.NA
    out["DistanceToSource"] = 0.0
    out["AttributeContinuity"] = 1.0
    out["ValidContinuityAttributeCount"] = len(ATTRIBUTE_COLUMNS)
    out["SourceDensity"] = safe_numeric(out["Density"])
    out["DensitySourceLogic"] = "step4_expert_real_well_prediction"
    out["DistanceConfidenceWeight"] = 1.0
    out["ConfidenceLogic"] = "real_well_hard_supervision"
    out["SingleSourceCheck"] = "pass"
    out["IsRefinedFracturePoint"] = out["SampleID"].isin(refined_point_ids).astype(int)
    out = normalize_density_columns(out)
    out = out[out["LayerGroup"].isin(ALLOWED_LAYERS)].copy()
    return add_missing_output_columns(out)


def normalize_virtual_samples(virtual_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        virtual_df,
        ["SampleID", "SourceSampleID", "SourceWellName", "VirtualWellName", "X", "Y", "TIME", "Density", "HasFracture", "PointConfidence", "StrataName"],
        "virtual well samples",
    )
    out = virtual_df.copy()
    out["SampleID"] = out["SampleID"].astype(str)
    out["SourceKind"] = "virtual_well"
    out["SourceWellName"] = out["SourceWellName"].astype(str)
    out["TrackWellName"] = out["VirtualWellName"].astype(str)
    out["WellName"] = out["VirtualWellName"].astype(str)
    out["LayerGroup"] = layer_group_from_strata(out["StrataName"])
    out["PointConfidence"] = safe_numeric(out["PointConfidence"]).clip(lower=0, upper=1)
    out["SingleSourceCheck"] = "pass"
    out["IsRefinedFracturePoint"] = 0
    out = normalize_density_columns(out)
    out = out[out["LayerGroup"].isin(ALLOWED_LAYERS)].copy()
    return add_missing_output_columns(out)


def append_csv(df: pd.DataFrame, output_csv: Path, write_header: bool) -> None:
    mode = "w" if write_header else "a"
    encoding = "utf-8-sig" if write_header else "utf-8"
    df.to_csv(output_csv, mode=mode, header=write_header, index=False, encoding=encoding)


def density_stats(series: pd.Series) -> dict[str, float | int | None]:
    numeric = safe_numeric(series).dropna()
    if numeric.empty:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": int(numeric.count()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
    }


class SummaryAccumulator:
    def __init__(self) -> None:
        self.total_rows = 0
        self.source_kind_counts: dict[str, int] = {}
        self.layer_counts: dict[str, int] = {}
        self.source_wells: dict[str, set[str]] = {}
        self.track_wells: dict[str, set[str]] = {}
        self.density_values: list[pd.Series] = []
        self.density_label_non_null = 0
        self.has_fracture_sum = 0
        self.point_confidence_sum = 0.0
        self.point_confidence_count = 0
        self.attr_non_null = {attr: 0 for attr in ATTRIBUTE_COLUMNS}
        self.qc_rows: list[dict[str, Any]] = []

    def add(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.total_rows += int(len(df))
        for key, value in df["SourceKind"].value_counts(dropna=False).items():
            self.source_kind_counts[str(key)] = self.source_kind_counts.get(str(key), 0) + int(value)
        for key, value in df["LayerGroup"].value_counts(dropna=False).items():
            self.layer_counts[str(key)] = self.layer_counts.get(str(key), 0) + int(value)
        for source_kind, group in df.groupby("SourceKind", dropna=False):
            source_kind_str = str(source_kind)
            self.source_wells.setdefault(source_kind_str, set()).update(group["SourceWellName"].dropna().astype(str).unique().tolist())
            self.track_wells.setdefault(source_kind_str, set()).update(group["TrackWellName"].dropna().astype(str).unique().tolist())
        density = safe_numeric(df["Density"])
        self.density_values.append(density)
        self.density_label_non_null += int(df["DensityLabel"].notna().sum())
        self.has_fracture_sum += int(safe_numeric(df["HasFracture"]).fillna(0).sum())
        confidence = safe_numeric(df["PointConfidence"]).dropna()
        self.point_confidence_sum += float(confidence.sum())
        self.point_confidence_count += int(confidence.count())
        for attr in ATTRIBUTE_COLUMNS:
            self.attr_non_null[attr] += int(pd.to_numeric(df[attr], errors="coerce").notna().sum())

        grouped = (
            df.groupby(["SourceKind", "LayerGroup"], dropna=False)
            .agg(
                SampleRows=("SampleID", "size"),
                SourceWellCount=("SourceWellName", "nunique"),
                TrackWellCount=("TrackWellName", "nunique"),
                DensityMean=("Density", "mean"),
                DensityMax=("Density", "max"),
                HasFractureRows=("HasFracture", "sum"),
                PointConfidenceMean=("PointConfidence", "mean"),
            )
            .reset_index()
        )
        self.qc_rows.extend(grouped.to_dict(orient="records"))

    def density_summary(self) -> dict[str, float | int | None]:
        if not self.density_values:
            return density_stats(pd.Series(dtype=float))
        return density_stats(pd.concat(self.density_values, ignore_index=True))

    def qc_frame(self) -> pd.DataFrame:
        if not self.qc_rows:
            return pd.DataFrame()
        raw = pd.DataFrame(self.qc_rows)
        out = (
            raw.groupby(["SourceKind", "LayerGroup"], dropna=False)
            .agg(
                SampleRows=("SampleRows", "sum"),
                SourceWellCount=("SourceWellCount", "max"),
                TrackWellCount=("TrackWellCount", "max"),
                DensityMean=("DensityMean", "mean"),
                DensityMax=("DensityMax", "max"),
                HasFractureRows=("HasFractureRows", "sum"),
                PointConfidenceMean=("PointConfidenceMean", "mean"),
            )
            .reset_index()
        )
        return out

    def to_summary(self) -> dict[str, Any]:
        return {
            "total_rows": int(self.total_rows),
            "source_kind_counts": {key: int(value) for key, value in sorted(self.source_kind_counts.items())},
            "layer_counts": {key: int(value) for key, value in sorted(self.layer_counts.items())},
            "source_well_counts": {key: int(len(value)) for key, value in sorted(self.source_wells.items())},
            "track_well_counts": {key: int(len(value)) for key, value in sorted(self.track_wells.items())},
            "density_stats": self.density_summary(),
            "density_label_non_null_count": int(self.density_label_non_null),
            "has_fracture_sum": int(self.has_fracture_sum),
            "mean_point_confidence": float(self.point_confidence_sum / self.point_confidence_count) if self.point_confidence_count else None,
            "attribute_non_null_counts": {key: int(value) for key, value in sorted(self.attr_non_null.items())},
        }


def build_field_contract(real_columns: set[str], virtual_columns: set[str], output_csv: Path) -> pd.DataFrame:
    rows = []
    for column in OUTPUT_COLUMNS:
        rows.append(
            {
                "Field": column,
                "Required": column in REQUIRED_OUTPUT_COLUMNS,
                "PresentInRealInput": column in real_columns,
                "PresentInVirtualInput": column in virtual_columns,
                "PresentInUnified": True,
                "Rule": field_rule(column),
            }
        )
    contract = pd.DataFrame(rows)
    contract.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return contract


def field_rule(column: str) -> str:
    rules = {
        "SourceKind": "real_well or virtual_well",
        "SourceWellName": "real source well for both real and virtual samples",
        "TrackWellName": "real well name for real samples; virtual well name for virtual samples",
        "WellName": "track-facing well name",
        "LayerGroup": "must be 沙三段 or 沙四段",
        "Density": "main supervision target",
        "DensityLabel": "non-null training label derived from Density",
        "HasFracture": "auxiliary binary flag",
        "PointConfidence": "1.0 for real samples; Step 5 confidence for virtual samples",
        "DensitySourceLogic": "Step 4 real prediction or Step 5 single-source attribute continuity",
        "IsRefinedFracturePoint": "Step 4 refined point membership for real samples",
    }
    return rules.get(column, "carried or normalized from formal upstream output")


def read_refined_point_ids(path: Path) -> tuple[set[str], int]:
    if not path.exists():
        return set(), 0
    points = pd.read_csv(path, usecols=["SourceSampleID"])
    ids = set(points["SourceSampleID"].dropna().astype(str).tolist())
    return ids, int(len(points))


def validate_virtual_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "single_source_bad_count": None}
    summary = pd.read_csv(path)
    bad = int((summary.get("SingleSourceCheck", pd.Series(index=summary.index, dtype=object)) != "pass").sum())
    return {
        "exists": True,
        "row_count": int(len(summary)),
        "virtual_well_count": int(summary["VirtualWellName"].nunique()) if "VirtualWellName" in summary.columns else None,
        "single_source_bad_count": bad,
    }


def validate_summary(summary: dict[str, Any], virtual_summary_check: dict[str, Any], contract: pd.DataFrame) -> dict[str, bool]:
    return {
        "has_rows": int(summary["total_rows"]) > 0,
        "has_real_and_virtual": {"real_well", "virtual_well"}.issubset(set(summary["source_kind_counts"].keys())),
        "density_label_non_null": int(summary["density_label_non_null_count"]) == int(summary["total_rows"]),
        "layers_limited_to_sha3_sha4": set(summary["layer_counts"].keys()).issubset(set(ALLOWED_LAYERS)),
        "required_fields_present": bool(contract.loc[contract["Required"], "PresentInUnified"].all()),
        "virtual_single_source_pass": virtual_summary_check.get("single_source_bad_count") in (0, None),
        "has_attribute_coverage": all(int(value) > 0 for value in summary["attribute_non_null_counts"].values()),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = read_json(config_path)
    real_csv = Path(config["real_well_prediction_csv"])
    points_csv = Path(config["real_fracture_points_csv"])
    virtual_csv = Path(config["virtual_well_training_samples_csv"])
    virtual_summary_csv = Path(config.get("virtual_well_training_summary_csv", ""))
    output_dir = Path(config["output_dir"])
    chunksize = int(config.get("chunksize", 250000))
    ensure_dir(output_dir)

    output_csv = output_dir / "unified_t4_t7_density_samples.csv"
    summary_json = output_dir / "unified_t4_t7_density_samples_summary.json"
    field_contract_csv = output_dir / "unified_t4_t7_field_contract.csv"
    qc_csv = output_dir / "unified_t4_t7_qc.csv"
    tmp_paths = {
        "samples": output_csv.with_name(f"{output_csv.name}.tmp"),
        "summary": summary_json.with_name(f"{summary_json.name}.tmp"),
        "contract": field_contract_csv.with_name(f"{field_contract_csv.name}.tmp"),
        "qc": qc_csv.with_name(f"{qc_csv.name}.tmp"),
    }

    refined_point_ids, refined_point_count = read_refined_point_ids(points_csv)
    accumulator = SummaryAccumulator()

    real_df = pd.read_csv(real_csv)
    real_columns = set(real_df.columns)
    real_out = normalize_real_samples(real_df, refined_point_ids)
    append_csv(real_out, tmp_paths["samples"], write_header=True)
    accumulator.add(real_out)
    print(f"[step6] real rows={len(real_out)}", flush=True)

    virtual_columns: set[str] | None = None
    write_header = False
    virtual_rows = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(virtual_csv, chunksize=chunksize), start=1):
        if virtual_columns is None:
            virtual_columns = set(chunk.columns)
        virtual_out = normalize_virtual_samples(chunk)
        append_csv(virtual_out, tmp_paths["samples"], write_header=write_header)
        accumulator.add(virtual_out)
        virtual_rows += int(len(virtual_out))
        print(f"[step6] virtual chunk={chunk_idx} virtual_rows={virtual_rows}", flush=True)
    if virtual_columns is None:
        virtual_columns = set()

    contract = build_field_contract(real_columns, virtual_columns, tmp_paths["contract"])
    qc_frame = accumulator.qc_frame()
    qc_frame.to_csv(tmp_paths["qc"], index=False, encoding="utf-8-sig")
    virtual_summary_check = validate_virtual_summary(virtual_summary_csv)
    summary = accumulator.to_summary()
    checks = validate_summary(summary, virtual_summary_check, contract)
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(config_path),
        "real_well_prediction_csv": str(real_csv),
        "real_fracture_points_csv": str(points_csv),
        "virtual_well_training_samples_csv": str(virtual_csv),
        "output_csv": str(output_csv),
        "field_contract_csv": str(field_contract_csv),
        "qc_csv": str(qc_csv),
        "allowed_layers": ALLOWED_LAYERS,
        "refined_fracture_point_count": int(refined_point_count),
        "virtual_summary_check": virtual_summary_check,
        "summary": summary,
        "checks": checks,
    }
    write_json(tmp_paths["summary"], payload)

    for key, tmp_path in tmp_paths.items():
        final_path = {
            "samples": output_csv,
            "summary": summary_json,
            "contract": field_contract_csv,
            "qc": qc_csv,
        }[key]
        tmp_path.replace(final_path)
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
