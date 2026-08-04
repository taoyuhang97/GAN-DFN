from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NULL_SENTINELS = (-99999, -99999.0, -9999, -9999.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified point-level well trajectory sample table.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON config file.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = exc
    raise RuntimeError(f"Failed to read csv: {path}") from last_error


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_null_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace(list(NULL_SENTINELS), np.nan)
    return out


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_non_empty_text(df: pd.DataFrame, columns: list[str], default: str = "") -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for column in columns:
        if column not in df.columns:
            continue
        candidate = df[column].copy()
        candidate = candidate.where(candidate.notna(), pd.NA)
        candidate = candidate.astype("string").str.strip()
        candidate = candidate.mask(candidate.eq(""), pd.NA)
        result = result.fillna(candidate)
    return result.fillna(default).astype(str)


def first_non_null_numeric(df: pd.DataFrame, columns: list[str], default: float = math.nan) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype=np.float64)
    for column in columns:
        if column not in df.columns:
            continue
        result = result.fillna(safe_numeric(df[column]))
    return result.fillna(default)


def join_unique_text(values: pd.Series) -> str:
    texts: list[str] = []
    seen: set[str] = set()
    for value in values.fillna("").astype(str):
        text = value.strip()
        if not text or text in seen:
            continue
        texts.append(text)
        seen.add(text)
    return ",".join(texts)


def circular_mean_deg(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    radians = np.deg2rad(values)
    sin_mean = np.mean(np.sin(radians))
    cos_mean = np.mean(np.cos(radians))
    if math.isclose(sin_mean, 0.0, abs_tol=1e-12) and math.isclose(cos_mean, 0.0, abs_tol=1e-12):
        return float("nan")
    angle = math.degrees(math.atan2(sin_mean, cos_mean)) % 360.0
    return float(angle)


def build_merge_key(df: pd.DataFrame, digits: int = 6) -> pd.Series:
    work_df = df.copy()
    key_cols: list[str] = []
    for source_col, key_name in [
        ("TVD", "__k_depth"),
        ("TIME", "__k_time"),
        ("X", "__k_x"),
        ("Y", "__k_y"),
    ]:
        if source_col not in work_df.columns:
            work_df[key_name] = ""
        else:
            numeric = safe_numeric(work_df[source_col]).round(digits)
            work_df[key_name] = numeric.map(
                lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
            )
        key_cols.append(key_name)
    if "WellName" in work_df.columns:
        work_df["__k_well"] = work_df["WellName"].fillna("").astype(str).str.strip()
    else:
        work_df["__k_well"] = ""
    key_cols.insert(0, "__k_well")
    return work_df[key_cols].agg("|".join, axis=1)


def merge_around_data(base_df: pd.DataFrame, around_df: pd.DataFrame) -> pd.DataFrame:
    left = base_df.sort_values("TVD").reset_index(drop=True).copy()
    right = around_df.sort_values("TVD").reset_index(drop=True).copy()
    around_keep_cols = [
        "TVD",
        "TIME",
        "X",
        "Y",
        "DEPT",
        "AC",
        "DEN",
        "GR",
        "RHOB",
        "SP",
    ]
    right = right[[col for col in around_keep_cols if col in right.columns]].copy()
    right = right.rename(
        columns={
            "TVD": "AroundTVD",
            "TIME": "AroundTIME",
            "X": "AroundX",
            "Y": "AroundY",
            "DEPT": "AroundDEPT",
            "AC": "AroundAC",
            "DEN": "AroundDEN",
            "GR": "AroundGR",
            "RHOB": "AroundRHOB",
            "SP": "AroundSP",
        }
    )
    merged = pd.merge_asof(
        left,
        right,
        left_on="TVD",
        right_on="AroundTVD",
        direction="nearest",
    )
    merged["AroundMatchDistance"] = (safe_numeric(merged["TVD"]) - safe_numeric(merged["AroundTVD"])).abs()
    merged["AroundMatchFlag"] = merged["AroundTVD"].notna().astype(int)
    merged["AttrAC"] = safe_numeric(merged.get("AroundAC", pd.Series(np.nan, index=merged.index)))
    merged["AttrDEN"] = safe_numeric(merged.get("AroundDEN", pd.Series(np.nan, index=merged.index)))
    merged["AttrGR"] = safe_numeric(merged.get("AroundGR", pd.Series(np.nan, index=merged.index)))
    merged["AttrRHOB"] = safe_numeric(merged.get("AroundRHOB", pd.Series(np.nan, index=merged.index)))
    merged["AttrSP"] = safe_numeric(merged.get("AroundSP", pd.Series(np.nan, index=merged.index)))
    return merged


def attach_strata_context(base_df: pd.DataFrame, strata_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.copy()
    if strata_df.empty:
        out["StrataMatchFlag"] = 0
        return out

    work = strata_df.copy()
    work["GeoDepthMin"] = safe_numeric(work["GeoDepthMin"])
    work["GeoDepthMax"] = safe_numeric(work["GeoDepthMax"])
    work = work[work["GeoDepthMin"].notna() & work["GeoDepthMax"].notna()].copy()
    work = work.sort_values(["GeoDepthMin", "GeoDepthMax", "GeoRangeOrder", "GeoSegmentID"]).reset_index(drop=True)
    starts = work["GeoDepthMin"].to_numpy(dtype=np.float64)
    ends = work["GeoDepthMax"].to_numpy(dtype=np.float64)
    depths = safe_numeric(out["TVD"]).to_numpy(dtype=np.float64)
    idx = np.searchsorted(starts, depths, side="right") - 1
    valid = (idx >= 0) & (idx < len(work))
    end_lookup = np.full(len(out), np.nan, dtype=np.float64)
    end_lookup[valid] = ends[idx[valid]]
    valid = valid & (depths <= (end_lookup + 1e-9))
    idx = np.where(valid, idx, -1)

    out["StrataMatchFlag"] = (idx >= 0).astype(int)
    mapping = {
        "WellName": "StrataWellName",
        "GeoSegmentID": "StrataGeoSegmentID",
        "GeoIntervalKey": "StrataGeoIntervalKey",
        "GeoRangeOrder": "StrataGeoRangeOrder",
        "GeoDepthMin": "StrataGeoDepthMin",
        "GeoDepthMax": "StrataGeoDepthMax",
        "StrataName": "StrataName",
        "TopSurfaceCode": "StrataTopSurfaceCode",
        "TopSurfaceName": "StrataTopSurfaceName",
        "BaseSurfaceCode": "StrataBaseSurfaceCode",
        "BaseSurfaceName": "StrataBaseSurfaceName",
        "StrataIntervalSource": "StrataIntervalSource",
        "ProvidedStrataName": "StrataProvidedName",
        "StrataAssignmentSource": "StrataAssignmentSource",
        "StrataAssignmentTopKMeanSimilarity": "StrataAssignmentTopKMeanSimilarity",
        "StrataAssignmentBestSimilarityScore": "StrataAssignmentBestSimilarityScore",
        "StrataAssignmentScoreMargin": "StrataAssignmentScoreMargin",
        "StrataAssignmentBestExpertWell": "StrataAssignmentBestExpertWell",
        "StrataAssignmentResolvedStrataName": "StrataAssignmentResolvedName",
        "PredSelectedExpertWell": "StrataPredSelectedExpertWell",
        "PredSelectedSimilarityScore": "StrataPredSelectedSimilarityScore",
        "PredSelectedSimilarityScoreSeismic": "StrataPredSelectedSimilarityScoreSeismic",
        "PredSelectedSimilarityScoreAC": "StrataPredSelectedSimilarityScoreAC",
        "PredSelectedSimilarityScoreGR": "StrataPredSelectedSimilarityScoreGR",
        "PredSelectionJointScore": "StrataPredSelectionJointScore",
        "PredSelectionQualityQualified": "StrataPredSelectionQualityQualified",
        "PredStage1InnerF1": "StrataPredStage1InnerF1",
        "PredStage2InnerCountErrorPct": "StrataPredStage2InnerCountErrorPct",
    }
    numeric_targets = {
        "StrataGeoRangeOrder",
        "StrataGeoDepthMin",
        "StrataGeoDepthMax",
        "StrataAssignmentTopKMeanSimilarity",
        "StrataAssignmentBestSimilarityScore",
        "StrataAssignmentScoreMargin",
        "StrataPredSelectedSimilarityScore",
        "StrataPredSelectedSimilarityScoreSeismic",
        "StrataPredSelectedSimilarityScoreAC",
        "StrataPredSelectedSimilarityScoreGR",
        "StrataPredSelectionJointScore",
        "StrataPredStage1InnerF1",
        "StrataPredStage2InnerCountErrorPct",
    }
    bool_targets = {"StrataPredSelectionQualityQualified"}

    for source_col, target_col in mapping.items():
        if source_col not in work.columns:
            out[target_col] = np.nan if target_col in numeric_targets else ""
            continue
        source_values = work[source_col].to_numpy(dtype=object)
        values = np.full(len(out), np.nan if target_col in numeric_targets else "", dtype=object)
        mask = idx >= 0
        if np.any(mask):
            values[mask] = source_values[idx[mask]]
        out[target_col] = values
        if target_col in numeric_targets:
            out[target_col] = safe_numeric(out[target_col])
        elif target_col in bool_targets:
            out[target_col] = out[target_col].map(lambda value: bool(value) if pd.notna(value) and value != "" else False)
    return out


def attach_final_point_details(base_df: pd.DataFrame, point_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.copy()
    default_numeric = {
        "PredPointCountAtDepth": 0,
        "PredPointDepthDistance": np.nan,
        "PredPointSegmentCount": 0,
        "PredPointDensityMassPerLength": np.nan,
        "PredPointDensityMass": np.nan,
        "PredPointAzimuth": np.nan,
        "PredPointDip": np.nan,
        "PredPointOrientationConfidence": np.nan,
    }
    default_text = {
        "PredPointSegmentIDs": "",
        "PredPointOrientationFamily": "",
    }
    out["PredPointMatchFlag"] = 0
    for col, value in default_numeric.items():
        out[col] = value
    for col, value in default_text.items():
        out[col] = value

    if point_df.empty:
        return out

    work = point_df.copy()
    base_depths = safe_numeric(out["TVD"]).to_numpy(dtype=np.float64)
    point_depths = safe_numeric(work["TVD"]).to_numpy(dtype=np.float64)
    grouped: dict[int, list[dict[str, Any]]] = {}

    for row, point_depth in zip(work.to_dict(orient="records"), point_depths):
        if not np.isfinite(point_depth):
            continue
        insert_idx = int(np.searchsorted(base_depths, point_depth, side="left"))
        if insert_idx <= 0:
            nearest_idx = 0
        elif insert_idx >= len(base_depths):
            nearest_idx = len(base_depths) - 1
        else:
            left_idx = insert_idx - 1
            right_idx = insert_idx
            nearest_idx = (
                left_idx
                if abs(base_depths[left_idx] - point_depth) <= abs(base_depths[right_idx] - point_depth)
                else right_idx
            )
        payload = dict(row)
        payload["__distance"] = abs(base_depths[nearest_idx] - point_depth)
        grouped.setdefault(nearest_idx, []).append(payload)

    for row_idx, rows in grouped.items():
        rows_df = pd.DataFrame(rows)
        out.at[row_idx, "PredPointCountAtDepth"] = int(len(rows_df))
        out.at[row_idx, "PredPointDepthDistance"] = float(
            safe_numeric(rows_df["__distance"]).min()
        )
        out.at[row_idx, "PredPointSegmentIDs"] = join_unique_text(rows_df["Segment_ID"])
        out.at[row_idx, "PredPointSegmentCount"] = int(rows_df["Segment_ID"].nunique())
        out.at[row_idx, "PredPointDensityMassPerLength"] = float(
            safe_numeric(rows_df["PointDensityMassPerLengthAllocated"]).sum()
        )
        out.at[row_idx, "PredPointDensityMass"] = float(
            safe_numeric(rows_df["PointDensityMassAllocated"]).sum()
        )
        out.at[row_idx, "PredPointAzimuth"] = float(
            safe_numeric(rows_df["PointAzimuth"]).mean()
        )
        out.at[row_idx, "PredPointDip"] = float(
            safe_numeric(rows_df["PointDip"]).mean()
        )
        out.at[row_idx, "PredPointOrientationFamily"] = join_unique_text(rows_df["PredOrientationFamily"])
        out.at[row_idx, "PredPointOrientationConfidence"] = float(
            safe_numeric(rows_df["PredOrientationConfidence"]).mean()
        )
    out["PredPointMatchFlag"] = safe_numeric(out["PredPointCountAtDepth"]).fillna(0).gt(0).astype(int)
    return out


def attach_segment_details(base_df: pd.DataFrame, segment_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.copy()
    out["PredSegmentMatchFlag"] = 0
    segment_mapping = {
        "Segment_ID": "PredSegmentIDInterval",
        "SegStartDepth": "PredSegmentStartDepth",
        "SegEndDepth": "PredSegmentEndDepth",
        "SegLength": "PredSegmentLength",
        "PredPointCount": "PredSegmentPointCount",
        "PredDensityStrength": "PredSegmentDensityStrength",
        "PredP10MassPerLength": "PredSegmentP10MassPerLength",
        "PredP10Mass": "PredSegmentP10Mass",
        "PredDensityStrengthLevel": "PredSegmentDensityLevel",
        "PredOrientationFamily": "PredSegmentOrientationFamily",
        "PredOrientationConfidence": "PredSegmentOrientationConfidence",
        "PredAzimuth": "PredSegmentAzimuth",
        "PredDip": "PredSegmentDip",
    }
    for target_col in segment_mapping.values():
        out[target_col] = np.nan

    if segment_df.empty:
        return out

    work = segment_df.copy()
    work["SegStartDepth"] = safe_numeric(work["SegStartDepth"])
    work["SegEndDepth"] = safe_numeric(work["SegEndDepth"])
    work = work[work["SegStartDepth"].notna() & work["SegEndDepth"].notna()].copy()
    work = work.sort_values(["SegStartDepth", "SegEndDepth", "Segment_ID"]).reset_index(drop=True)
    starts = work["SegStartDepth"].to_numpy(dtype=np.float64)
    ends = work["SegEndDepth"].to_numpy(dtype=np.float64)
    depths = safe_numeric(out["TVD"]).to_numpy(dtype=np.float64)
    idx = np.searchsorted(starts, depths, side="right") - 1
    valid = (idx >= 0) & (idx < len(work))
    end_lookup = np.full(len(out), np.nan, dtype=np.float64)
    end_lookup[valid] = ends[idx[valid]]
    valid = valid & (depths <= (end_lookup + 1e-9))
    idx = np.where(valid, idx, -1)
    out["PredSegmentMatchFlag"] = (idx >= 0).astype(int)
    for source_col, target_col in segment_mapping.items():
        source_values = work[source_col].to_numpy(dtype=object) if source_col in work.columns else np.array([], dtype=object)
        values = np.full(len(out), np.nan, dtype=object)
        mask = idx >= 0
        if source_values.size > 0 and np.any(mask):
            values[mask] = source_values[idx[mask]]
        out[target_col] = values
        if target_col in {"PredSegmentDensityLevel", "PredSegmentOrientationFamily"}:
            out[target_col] = out[target_col].fillna("").astype(str)
        else:
            out[target_col] = safe_numeric(out[target_col])
    return out


def attach_raw_fractures(base_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.copy()
    out["RawFracturePointFlag"] = 0
    out["RawFracturePointCount"] = 0
    out["RawFractureNearestMD"] = np.nan
    out["RawFractureDepthDistance"] = np.nan
    out["RawFractureAzimuth"] = np.nan
    out["RawFractureDip"] = np.nan

    if raw_df.empty or out.empty:
        return out

    depths = safe_numeric(out["TVD"]).to_numpy(dtype=np.float64)
    valid_sample_mask = np.isfinite(depths)
    if not np.all(valid_sample_mask):
        out = out.loc[valid_sample_mask].reset_index(drop=True)
        depths = safe_numeric(out["TVD"]).to_numpy(dtype=np.float64)

    min_depth = float(depths[0])
    max_depth = float(depths[-1])
    grouped: dict[int, list[dict[str, float]]] = {}
    raw_points = raw_df.rename(
        columns={
            "Azimuth(0~360)": "RawFracAzimuth",
            "Angle(0~90)": "RawFracDip",
        }
    )

    for row in raw_points.itertuples(index=False):
        point_depth = float(getattr(row, "MD"))
        if not np.isfinite(point_depth) or point_depth < min_depth or point_depth > max_depth:
            continue
        insert_idx = int(np.searchsorted(depths, point_depth, side="left"))
        if insert_idx <= 0:
            nearest_idx = 0
        elif insert_idx >= len(depths):
            nearest_idx = len(depths) - 1
        else:
            left_idx = insert_idx - 1
            right_idx = insert_idx
            nearest_idx = (
                left_idx
                if abs(depths[left_idx] - point_depth) <= abs(depths[right_idx] - point_depth)
                else right_idx
            )
        grouped.setdefault(nearest_idx, []).append(
            {
                "MD": point_depth,
                "RawFracAzimuth": float(getattr(row, "RawFracAzimuth")),
                "RawFracDip": float(getattr(row, "RawFracDip")),
                "Distance": abs(depths[nearest_idx] - point_depth),
            }
        )

    for row_idx, rows in grouped.items():
        azimuths = np.asarray([item["RawFracAzimuth"] for item in rows], dtype=np.float64)
        dips = np.asarray([item["RawFracDip"] for item in rows], dtype=np.float64)
        distances = np.asarray([item["Distance"] for item in rows], dtype=np.float64)
        mds = np.asarray([item["MD"] for item in rows], dtype=np.float64)
        out.at[row_idx, "RawFracturePointFlag"] = 1
        out.at[row_idx, "RawFracturePointCount"] = int(len(rows))
        out.at[row_idx, "RawFractureNearestMD"] = float(np.mean(mds)) if mds.size else np.nan
        out.at[row_idx, "RawFractureDepthDistance"] = float(np.min(distances)) if distances.size else np.nan
        out.at[row_idx, "RawFractureAzimuth"] = circular_mean_deg(azimuths)
        out.at[row_idx, "RawFractureDip"] = float(np.mean(dips)) if dips.size else np.nan
    return out


def finalize_table(
    config_name: str,
    config: dict[str, Any],
    base_df: pd.DataFrame,
) -> pd.DataFrame:
    out = base_df.copy()
    source_inputs = config["inputs"]
    out.insert(0, "UnifiedRowID", np.arange(1, len(out) + 1, dtype=np.int64))
    out.insert(1, "ConfigName", config_name)
    out.insert(2, "UnifiedSampleVersion", "step1_minimal_v1")
    out.insert(3, "BaseRowSource", "final_log_with_fractures")
    out.insert(4, "SourceAroundFile", Path(source_inputs["around_data_csv"]).name)
    out.insert(5, "SourceRawFractureFile", Path(source_inputs["fractures_csv"]).name)
    out.insert(6, "SourceFinalLogFile", Path(source_inputs["final_log_with_fractures_csv"]).name)
    out.insert(7, "SourceFinalPointFile", Path(source_inputs["final_fracture_points_csv"]).name)
    out.insert(8, "SourceFinalSegmentFile", Path(source_inputs["final_fracture_segments_csv"]).name)
    out.insert(9, "SourceStrataFile", Path(source_inputs["final_strata_segmentation_csv"]).name)
    out["CommonFractureDensity"] = safe_numeric(out.get("PredDensityMassPerLength", pd.Series(np.nan, index=out.index)))
    out["CommonFractureAzimuth"] = safe_numeric(out.get("PredAzimuth", pd.Series(np.nan, index=out.index)))
    out["CommonFractureDip"] = safe_numeric(out.get("PredDip", pd.Series(np.nan, index=out.index)))
    out["CommonFractureFlag"] = safe_numeric(out.get("PredFractureFlag", pd.Series(0, index=out.index))).fillna(0).astype(int)
    out["CommonStrataName"] = out.get("StrataName", pd.Series("", index=out.index)).fillna("").astype(str)
    out["SourceWellName"] = first_non_empty_text(
        out,
        ["WellName", "StrataWellName"],
        default=str(config.get("well_name", "UNKNOWN_WELL")),
    )
    out["SourceType"] = str(config.get("source_type", "well_trajectory_point_mvp"))
    out["TVDRef"] = first_non_null_numeric(out, ["TVD", "DEPT"])
    out["LayerGroup"] = first_non_empty_text(
        out,
        [
            "StrataAssignmentResolvedName",
            "CommonStrataName",
            "StrataName",
            "InterpStrataName",
            "StrataProvidedName",
        ],
        default="UNSPECIFIED",
    )
    out["FractureDensity"] = first_non_null_numeric(
        out,
        ["CommonFractureDensity", "PredDensityMassPerLength", "PredPointDensityMassPerLength"],
        default=0.0,
    )
    out["IsFracturePoint"] = (
        (safe_numeric(out["CommonFractureFlag"]).fillna(0) > 0)
        | (safe_numeric(out["FractureDensity"]).fillna(0) > 0)
    ).astype(int)
    out["PointConfidence"] = np.where(
        out["IsFracturePoint"].eq(1),
        first_non_null_numeric(
            out,
            ["PredPointOrientationConfidence", "PredSegmentOrientationConfidence"],
            default=1.0,
        ),
        0.0,
    )
    out["MissingGTDensity"] = np.nan
    out["MissingGTDensityNote"] = "not_available_in_required_step1_inputs"
    out["MissingFaultConstraint"] = np.nan
    out["MissingFaultConstraintNote"] = "reserved_for_later_steps"
    front_cols = [
        "UnifiedRowID",
        "ConfigName",
        "UnifiedSampleVersion",
        "BaseRowSource",
        "SourceAroundFile",
        "SourceRawFractureFile",
        "SourceFinalLogFile",
        "SourceFinalPointFile",
        "SourceFinalSegmentFile",
        "SourceStrataFile",
        "SourceWellName",
        "SourceType",
        "TVDRef",
        "LayerGroup",
        "FractureDensity",
        "IsFracturePoint",
        "PointConfidence",
        "WellName",
        "TVD",
        "DEPT",
        "TIME",
        "X",
        "Y",
        "AttrAC",
        "AttrDEN",
        "AttrGR",
        "AttrRHOB",
        "AttrSP",
        "AroundMatchFlag",
        "AroundMatchDistance",
        "AroundTVD",
        "AroundTIME",
        "AroundX",
        "AroundY",
        "RawFracturePointFlag",
        "RawFracturePointCount",
        "RawFractureNearestMD",
        "RawFractureDepthDistance",
        "RawFractureAzimuth",
        "RawFractureDip",
        "CommonFractureFlag",
        "CommonFractureDensity",
        "CommonFractureAzimuth",
        "CommonFractureDip",
        "PredPointMatchFlag",
        "PredPointCountAtDepth",
        "PredPointSegmentIDs",
        "PredPointSegmentCount",
        "PredPointDensityMassPerLength",
        "PredPointDensityMass",
        "PredPointAzimuth",
        "PredPointDip",
        "PredPointOrientationFamily",
        "PredPointOrientationConfidence",
        "PredSegmentMatchFlag",
        "PredSegmentIDInterval",
        "PredSegmentStartDepth",
        "PredSegmentEndDepth",
        "PredSegmentLength",
        "PredSegmentPointCount",
        "PredSegmentDensityStrength",
        "PredSegmentP10MassPerLength",
        "PredSegmentP10Mass",
        "PredSegmentDensityLevel",
        "PredSegmentOrientationFamily",
        "PredSegmentOrientationConfidence",
        "PredSegmentAzimuth",
        "PredSegmentDip",
        "StrataMatchFlag",
        "CommonStrataName",
        "StrataGeoSegmentID",
        "StrataGeoIntervalKey",
        "StrataGeoRangeOrder",
        "StrataGeoDepthMin",
        "StrataGeoDepthMax",
        "StrataTopSurfaceCode",
        "StrataTopSurfaceName",
        "StrataBaseSurfaceCode",
        "StrataBaseSurfaceName",
        "StrataIntervalSource",
        "StrataAssignmentSource",
        "StrataAssignmentTopKMeanSimilarity",
        "StrataAssignmentBestSimilarityScore",
        "StrataAssignmentScoreMargin",
        "StrataAssignmentBestExpertWell",
        "StrataAssignmentResolvedName",
        "StrataPredSelectedExpertWell",
        "StrataPredSelectedSimilarityScore",
        "StrataPredSelectedSimilarityScoreSeismic",
        "StrataPredSelectedSimilarityScoreAC",
        "StrataPredSelectedSimilarityScoreGR",
        "StrataPredSelectionJointScore",
        "StrataPredSelectionQualityQualified",
        "StrataPredStage1InnerF1",
        "StrataPredStage2InnerCountErrorPct",
        "MissingGTDensity",
        "MissingGTDensityNote",
        "MissingFaultConstraint",
        "MissingFaultConstraintNote",
    ]
    ordered_front = [col for col in front_cols if col in out.columns]
    remaining_cols = [col for col in out.columns if col not in ordered_front]
    return out[ordered_front + remaining_cols]


def build_summary(config_name: str, config: dict[str, Any], unified_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "config_name": config_name,
        "well_name": config.get("well_name", ""),
        "row_count": int(len(unified_df)),
        "column_count": int(len(unified_df.columns)),
        "predicted_fracture_positive_rows": int(safe_numeric(unified_df["CommonFractureFlag"]).fillna(0).sum()),
        "raw_fracture_positive_rows": int(safe_numeric(unified_df["RawFracturePointFlag"]).fillna(0).sum()),
        "strata_matched_rows": int(safe_numeric(unified_df["StrataMatchFlag"]).fillna(0).sum()),
        "around_matched_rows": int(safe_numeric(unified_df["AroundMatchFlag"]).fillna(0).sum()),
        "core_fields": [
            "UnifiedRowID",
            "WellName",
            "TVD",
            "DEPT",
            "TIME",
            "X",
            "Y",
            "SourceWellName",
            "SourceType",
            "TVDRef",
            "LayerGroup",
            "FractureDensity",
            "IsFracturePoint",
            "PointConfidence",
            "AttrAC",
            "AttrDEN",
            "AttrGR",
            "AttrRHOB",
            "AttrSP",
            "CommonFractureFlag",
            "CommonFractureDensity",
            "RawFracturePointFlag",
            "RawFracturePointCount",
            "PredPointMatchFlag",
            "PredSegmentMatchFlag",
            "CommonStrataName",
            "StrataGeoSegmentID",
        ],
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    config_name = config_path.stem
    inputs = config["inputs"]

    around_df = normalize_null_sentinels(read_csv_flexible(Path(inputs["around_data_csv"])))
    raw_fracture_df = normalize_null_sentinels(read_csv_flexible(Path(inputs["fractures_csv"])))
    final_log_df = normalize_null_sentinels(read_csv_flexible(Path(inputs["final_log_with_fractures_csv"])))
    final_point_df = normalize_null_sentinels(read_csv_flexible(Path(inputs["final_fracture_points_csv"])))
    final_segment_df = normalize_null_sentinels(read_csv_flexible(Path(inputs["final_fracture_segments_csv"])))
    strata_df = normalize_null_sentinels(read_csv_flexible(Path(inputs["final_strata_segmentation_csv"])))

    base_df = final_log_df.copy()
    base_df["WellName"] = base_df.get("WellName", pd.Series(config.get("well_name", ""), index=base_df.index))
    base_df = base_df.sort_values(["TVD", "TIME"]).reset_index(drop=True)
    base_df = merge_around_data(base_df, around_df)
    base_df = attach_strata_context(base_df, strata_df)
    base_df = attach_final_point_details(base_df, final_point_df)
    base_df = attach_segment_details(base_df, final_segment_df)
    base_df = attach_raw_fractures(base_df, raw_fracture_df)
    unified_df = finalize_table(config_name=config_name, config=config, base_df=base_df)

    output_csv = Path(config["output_csv"]).resolve()
    ensure_parent(output_csv)
    unified_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = build_summary(config_name=config_name, config=config, unified_df=unified_df)
    summary_path = output_csv.with_name(f"{output_csv.stem}_summary.json")
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Built unified point table: {output_csv}")
    print(f"Rows={len(unified_df)} Cols={len(unified_df.columns)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
