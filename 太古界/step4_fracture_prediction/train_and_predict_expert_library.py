from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
STEP2_ROOT_DEFAULT = (
    REPO_ROOT / "优化阶段二/正式主线/step2_real_well_t4_t7_samples/output/formal_all_wells"
)
STEP3_GROUPS_DEFAULT = (
    REPO_ROOT / "优化阶段二/正式主线/step3_imaging_supervision_samples/output/formal_rebuild/groups"
)
TARGET_STRATA = ["沙三段", "沙四段"]
DEFAULT_HOLDOUT_WELL = "车页1导眼"
DEFAULT_FEATURE_COLUMNS = [
    "AC",
    "CAL",
    "CNL",
    "DEN",
    "GR",
    "RFOC",
    "RILD",
    "RILM",
    "SP",
    "SeisAmp",
    "Coherence",
    "AntTrack",
    "CurvatureMax",
    "CurvaturePos",
    "SeisAmpMean",
    "SeisAmpStd",
    "SeisAmpMin",
    "SeisAmpMax",
    "SeisAmpValidCount",
    "CoherenceMean",
    "CoherenceStd",
    "CoherenceMin",
    "CoherenceMax",
    "CoherenceValidCount",
    "AntTrackMean",
    "AntTrackStd",
    "AntTrackMin",
    "AntTrackMax",
    "AntTrackValidCount",
    "CurvatureMaxMean",
    "CurvatureMaxStd",
    "CurvatureMaxMin",
    "CurvatureMaxMax",
    "CurvatureMaxValidCount",
    "CurvaturePosMean",
    "CurvaturePosStd",
    "CurvaturePosMin",
    "CurvaturePosMax",
    "CurvaturePosValidCount",
]
IDENTITY_COLUMNS = ["SampleID", "WellName", "X", "Y", "TIME", "TVD", "DEPT", "StrataName"]


@dataclass(frozen=True)
class ExpertLibraryBundle:
    feature_columns: list[str]
    signature_columns: list[str]
    min_nonnull_features: int
    target_column: str
    stage1_threshold: float
    stage1_thresholds_by_strata: dict[str, float]
    stage1_thresholds_by_expert: dict[str, float]
    expert_quality_by_id: dict[str, dict[str, Any]]
    point_count_calibrators_by_strata: dict[str, dict[str, Any]]
    experts_by_strata: dict[str, dict[str, dict[str, Any]]]
    metadata: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Step 4 strata-first well-level expert library and predict all real wells."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--train-only", action="store_true", help="Only train the expert library.")
    parser.add_argument("--predict-only", action="store_true", help="Only run prediction with an existing bundle.")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_bool(series: pd.Series, default: bool = True) -> pd.Series:
    if series is None:
        return pd.Series(default)
    if series.dtype == bool:
        return series.fillna(default)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "yes": True, "usable": True, "false": False, "0": False, "no": False})
        .fillna(default)
        .astype(bool)
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def safe_quantile(values: pd.Series | np.ndarray, q: float) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def integrate_trapezoid(depth: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    n = len(depth)
    out = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return out
    dx = np.diff(depth)
    dx = np.clip(dx, a_min=0.0, a_max=None)
    area = 0.5 * (intensity[1:] + intensity[:-1]) * dx
    out[1:] = np.cumsum(area)
    return out


def integrate_total(depth: np.ndarray, intensity: np.ndarray | None) -> float:
    if intensity is None or len(depth) <= 1:
        return 0.0
    return float(integrate_trapezoid(depth, intensity)[-1])


def pick_points_by_equal_intensity(depth: np.ndarray, intensity: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    if n_points <= 0 or len(depth) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
    if len(depth) == 1:
        return np.array([float(depth[0])]), np.array([0], dtype=np.int64)
    n_points = int(min(n_points, len(depth)))
    cum = integrate_trapezoid(depth, intensity)
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 0.0:
        picked_idx = np.arange(n_points, dtype=np.int64)
        return depth[picked_idx].astype(np.float64), picked_idx
    step = total / n_points
    targets = (np.arange(n_points, dtype=np.float64) + 0.5) * step
    picked_depths = np.zeros(n_points, dtype=np.float64)
    picked_idx = np.zeros(n_points, dtype=np.int64)
    for i, target in enumerate(targets):
        idx = int(np.searchsorted(cum, target, side="left"))
        if idx <= 0:
            picked_depths[i] = float(depth[0])
            picked_idx[i] = 0
            continue
        if idx >= len(depth):
            picked_depths[i] = float(depth[-1])
            picked_idx[i] = len(depth) - 1
            continue
        d0, d1 = float(depth[idx - 1]), float(depth[idx])
        c0, c1 = float(cum[idx - 1]), float(cum[idx])
        if c1 <= c0:
            picked_depths[i] = d1
        else:
            ratio = (target - c0) / (c1 - c0)
            picked_depths[i] = d0 + ratio * (d1 - d0)
        picked_idx[i] = idx if abs(depth[idx] - picked_depths[i]) < abs(depth[idx - 1] - picked_depths[i]) else idx - 1
    return picked_depths, picked_idx


def compute_n_points(total_intensity: float, min_points: int, rounding_mode: str = "round") -> int:
    if not np.isfinite(total_intensity) or total_intensity <= 0.0:
        return 0
    if rounding_mode == "floor":
        n = int(np.floor(total_intensity))
    elif rounding_mode == "ceil":
        n = int(np.ceil(total_intensity))
    else:
        n = int(np.round(total_intensity))
    return max(int(min_points), n)


def find_binary_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif (not value) and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def compute_shape_intensity(
    density_seg: np.ndarray,
    prob_seg: np.ndarray,
    shape_mode: str = "density_prob",
    prob_weight_gamma: float = 2.0,
) -> np.ndarray:
    density_seg = np.clip(np.nan_to_num(density_seg, nan=0.0, posinf=0.0, neginf=0.0), a_min=0.0, a_max=None)
    prob_seg = np.clip(np.nan_to_num(prob_seg, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    prob_gamma = np.power(prob_seg, prob_weight_gamma)
    if shape_mode == "density_only":
        return density_seg
    if shape_mode == "prob_only":
        return prob_gamma
    return density_seg * prob_gamma


def extract_segment_payloads(
    pred_df: pd.DataFrame,
    threshold: float,
    min_density: float,
    shape_mode: str = "density_prob",
    prob_weight_gamma: float = 2.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    work = pred_df.copy()
    depth_col = "DEPT" if "DEPT" in work.columns else "TIME"
    sort_columns = [col for col in [depth_col, "TIME", "SampleID"] if col in work.columns]
    if sort_columns:
        work = work.sort_values(sort_columns, kind="mergesort").copy()

    depth = safe_numeric(work[depth_col]).to_numpy(dtype=np.float64)
    prob = safe_numeric(work["PredFractureProb"]).fillna(0.0).to_numpy(dtype=np.float64)
    density = safe_numeric(work["PredDensity"]).fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    mask = (prob >= float(threshold)) & (density >= float(min_density))

    payloads: list[dict[str, Any]] = []
    for seg_id, (start_idx, end_idx) in enumerate(find_binary_segments(mask), start=1):
        depth_seg = depth[start_idx : end_idx + 1]
        prob_seg = prob[start_idx : end_idx + 1]
        density_seg = density[start_idx : end_idx + 1]
        shape_intensity = compute_shape_intensity(
            density_seg=density_seg,
            prob_seg=prob_seg,
            shape_mode=shape_mode,
            prob_weight_gamma=prob_weight_gamma,
        )
        payloads.append(
            {
                "Segment_ID": seg_id,
                "StartIdx": int(start_idx),
                "EndIdx": int(end_idx),
                "SegStartDepth": float(depth_seg[0]) if len(depth_seg) else np.nan,
                "SegEndDepth": float(depth_seg[-1]) if len(depth_seg) else np.nan,
                "SegLength": float(max(depth_seg[-1] - depth_seg[0], 0.0)) if len(depth_seg) >= 2 else 0.0,
                "RawDensityTotal": integrate_total(depth_seg, density_seg),
                "ProbMass": integrate_total(depth_seg, np.power(prob_seg, prob_weight_gamma)),
                "ProbMean": float(np.mean(prob_seg)) if len(prob_seg) else np.nan,
                "DensityMean": float(np.mean(density_seg)) if len(density_seg) else np.nan,
                "depth_seg": depth_seg,
                "density_seg": density_seg,
                "shape_intensity": shape_intensity,
            }
        )
    return work, payloads, depth_col


def count_points_in_depth_range(point_depths: np.ndarray, start_depth: float, end_depth: float) -> int:
    if point_depths.size == 0:
        return 0
    return int(((point_depths >= start_depth) & (point_depths <= end_depth)).sum())


def fit_point_count_calibrator(segment_rows: pd.DataFrame, basis_col: str = "RawDensityTotal") -> dict[str, Any]:
    if segment_rows.empty:
        return {
            "basis_col": basis_col,
            "scale": 1.0,
            "basis_sum": np.nan,
            "target_sum": np.nan,
            "segment_count": 0,
            "well_count": 0,
            "shape_mode": "density_prob",
            "mode": "global_density_scale",
            "used_well_count": 0,
            "dropped_zero_cover_well_count": 0,
        }
    work = segment_rows.copy()
    if "WellSegment" in work.columns:
        well_gt = work.groupby("WellSegment")["GTPointCountInPredSegment"].sum().reset_index(name="WellGTPointSum")
        valid_wells = set(well_gt.loc[well_gt["WellGTPointSum"] > 0, "WellSegment"].astype(str))
        dropped_zero_cover_well_count = int((well_gt["WellGTPointSum"] <= 0).sum())
        if valid_wells:
            work = work[work["WellSegment"].astype(str).isin(valid_wells)].copy()
        used_well_count = len(valid_wells)
    else:
        dropped_zero_cover_well_count = 0
        used_well_count = 0
    if work.empty:
        return {
            "basis_col": basis_col,
            "scale": 1.0,
            "basis_sum": np.nan,
            "target_sum": np.nan,
            "segment_count": 0,
            "well_count": int(segment_rows["WellSegment"].astype(str).nunique()) if "WellSegment" in segment_rows.columns else 0,
            "shape_mode": "density_prob",
            "mode": "global_density_scale",
            "used_well_count": used_well_count,
            "dropped_zero_cover_well_count": dropped_zero_cover_well_count,
        }
    basis = safe_numeric(work[basis_col]).fillna(0.0).clip(lower=0.0)
    target = safe_numeric(work["GTPointCountInPredSegment"]).fillna(0.0).clip(lower=0.0)
    basis_sum = float(basis.sum())
    target_sum = float(target.sum())
    scale = float(target_sum / basis_sum) if basis_sum > 1e-8 else 1.0
    return {
        "basis_col": basis_col,
        "scale": scale,
        "basis_sum": basis_sum,
        "target_sum": target_sum,
        "segment_count": int(len(work)),
        "well_count": int(segment_rows["WellSegment"].astype(str).nunique()) if "WellSegment" in segment_rows.columns else 0,
        "shape_mode": "density_prob",
        "mode": "global_density_scale",
        "used_well_count": used_well_count,
        "dropped_zero_cover_well_count": dropped_zero_cover_well_count,
    }


def nearest_distance_stats(src_depths: np.ndarray, ref_depths: np.ndarray) -> dict[str, float]:
    if src_depths.size == 0 or ref_depths.size == 0:
        return {"mean": np.nan, "median": np.nan, "p90": np.nan, "max": np.nan}
    src = np.sort(src_depths.astype(np.float64))
    ref = np.sort(ref_depths.astype(np.float64))
    dists = np.zeros_like(src)
    ref_idx = 0
    for src_idx, depth in enumerate(src):
        while ref_idx + 1 < ref.size and abs(ref[ref_idx + 1] - depth) <= abs(ref[ref_idx] - depth):
            ref_idx += 1
        dists[src_idx] = abs(ref[ref_idx] - depth)
    dists_sorted = np.sort(dists)
    p90_idx = int(np.floor(0.9 * max(len(dists_sorted) - 1, 0)))
    return {
        "mean": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "p90": float(dists_sorted[p90_idx]),
        "max": float(np.max(dists)),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2), encoding="utf-8")


def safe_file_stem(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", text)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = df.copy()
    for column in columns:
        if column not in work.columns:
            work[column] = np.nan
    return work


def build_feature_mask(df: pd.DataFrame, feature_columns: list[str], min_nonnull_features: int) -> pd.Series:
    available = [col for col in feature_columns if col in df.columns]
    if not available:
        return pd.Series(False, index=df.index)
    return df[available].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1) >= int(min_nonnull_features)


def build_stage1_model(random_state: int, n_estimators: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=10,
                    min_samples_leaf=8,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_dummy_stage1_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )


def build_stage2_model(random_state: int, n_estimators: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=12,
                    min_samples_leaf=10,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def read_step3_groups(groups_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_paths = sorted(groups_dir.glob("*.csv"))
    if not group_paths:
        raise RuntimeError(f"No Step 3 group csv files found: {groups_dir}")
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for group_path in group_paths:
        df = pd.read_csv(group_path)
        group_id = group_path.stem
        if "WellSegment" not in df.columns or "StrataName" not in df.columns:
            raise RuntimeError(f"Step 3 group missing WellSegment/StrataName: {group_path}")
        well_segment = str(df["WellSegment"].dropna().iloc[0]) if df["WellSegment"].notna().any() else ""
        strata_name = str(df["StrataName"].dropna().iloc[0]) if df["StrataName"].notna().any() else ""
        df["GroupID"] = group_id
        df["SourceGroupFile"] = str(group_path)
        df["WellName"] = df["WellSegment"]
        frames.append(df)
        density = safe_numeric(df["Density"]) if "Density" in df.columns else pd.Series(np.nan, index=df.index)
        manifest_rows.append(
            {
                "GroupID": group_id,
                "WellSegment": well_segment,
                "StrataName": strata_name,
                "Path": str(group_path),
                "RowCount": int(len(df)),
                "DensityNonNullRows": int(density.notna().sum()),
                "DensityPositiveRows": int((density.fillna(0.0) > 0.0).sum()),
                "SampleUsableRows": int(safe_bool(df.get("SampleUsableForModel", pd.Series(True, index=df.index))).sum()),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(manifest_rows)


def discover_step2_wells(step2_root: Path) -> list[str]:
    return sorted(path.name for path in step2_root.iterdir() if path.is_dir())


def read_step2_summary(step2_root: Path) -> pd.DataFrame:
    summary_csv = step2_root / "real_well_t4_t7_summary.csv"
    if not summary_csv.exists():
        rows = []
        for well_name in discover_step2_wells(step2_root):
            well_dir = step2_root / well_name
            rows.append(
                {
                    "WellName": well_name,
                    "Status": "ok",
                    "MainCsv": str(well_dir / f"{well_name}_t4_t7_real_well_main.csv"),
                    "IntervalCsv": str(well_dir / f"{well_name}_t4_t7_real_well_interval.csv"),
                }
            )
        return pd.DataFrame(rows)
    return pd.read_csv(summary_csv)


def step2_ok_wells(step2_root: Path) -> pd.DataFrame:
    summary = read_step2_summary(step2_root)
    if "Status" not in summary.columns:
        raise RuntimeError(f"Step2 summary missing Status column under {step2_root}")
    ok = summary[summary["Status"].astype(str) == "ok"].copy()
    required = ["WellName", "MainCsv", "IntervalCsv"]
    missing = [col for col in required if col not in ok.columns]
    if missing:
        raise RuntimeError(f"Step2 summary missing required columns: {missing}")
    return ok


def load_step2_main_for_feature_scan(step2_root: Path, wells: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for well_name in wells:
        main_csv = step2_root / well_name / f"{well_name}_t4_t7_real_well_main.csv"
        if main_csv.exists():
            frames.append(pd.read_csv(main_csv, nrows=2000))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def select_feature_columns(
    config: dict[str, Any],
    train_source_df: pd.DataFrame,
    target_probe_df: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    configured = list(config.get("feature_columns") or DEFAULT_FEATURE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for column in configured:
        in_step3 = column in train_source_df.columns
        in_step2 = column in target_probe_df.columns
        train_nonnull = int(safe_numeric(train_source_df[column]).notna().sum()) if in_step3 else 0
        target_nonnull = int(safe_numeric(target_probe_df[column]).notna().sum()) if in_step2 else 0
        rows.append(
            {
                "FeatureColumn": column,
                "Configured": True,
                "InStep3Groups": bool(in_step3),
                "InStep2Targets": bool(in_step2),
                "Step3NonNullRows": train_nonnull,
                "Step2ProbeNonNullRows": target_nonnull,
                "UseForModel": True,
                "Role": "formal_model_feature",
            }
        )
    if not configured:
        raise RuntimeError("No configured Step 4 feature columns.")
    return configured, pd.DataFrame(rows)


def usable_training_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    min_nonnull_features: int,
    target_column: str,
) -> pd.DataFrame:
    work = ensure_columns(df, [*feature_columns, target_column, "HasFracture"])
    for column in [*feature_columns, target_column, "HasFracture"]:
        work[column] = safe_numeric(work[column])
    usable_col = work.get("SampleUsableForModel", pd.Series(True, index=work.index))
    mask = (
        safe_bool(usable_col, default=True).reindex(work.index, fill_value=True)
        & work["StrataName"].isin(TARGET_STRATA)
        & work[target_column].notna()
        & build_feature_mask(work, feature_columns, min_nonnull_features)
    )
    return work[mask].copy()


def expert_feature_columns(df: pd.DataFrame, feature_columns: list[str], min_nonnull_features: int) -> list[str]:
    cols: list[str] = []
    for column in feature_columns:
        if column not in df.columns:
            continue
        values = safe_numeric(df[column])
        if values.notna().sum() == 0:
            continue
        if values.notna().sum() < int(min_nonnull_features):
            continue
        cols.append(column)
    return cols


def train_one_expert(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    random_state: int,
    cls_estimators: int,
    reg_estimators: int,
) -> tuple[Pipeline, Pipeline, dict[str, Any], list[str]]:
    model_feature_columns = list(feature_columns)
    x = ensure_columns(train_df, model_feature_columns)[model_feature_columns].apply(pd.to_numeric, errors="coerce")
    density = safe_numeric(train_df[target_column]).fillna(0.0).clip(lower=0.0)
    y_cls = (density > 0.0).astype(int)
    y_reg = density.to_numpy(dtype=float)
    if y_cls.nunique(dropna=True) < 2:
        cls_model = build_dummy_stage1_model()
    else:
        cls_model = build_stage1_model(random_state, cls_estimators)
    reg_model = build_stage2_model(random_state, reg_estimators)
    cls_model.fit(x, y_cls)
    reg_model.fit(x, y_reg)
    return cls_model, reg_model, {
        "ClassCount": int(y_cls.nunique(dropna=True)),
        "PositiveDensityRows": int((density > 0.0).sum()),
        "MeanDensity": float(density.mean()) if len(density) else np.nan,
    }, model_feature_columns


def calibrate_stage1_threshold(
    actual_density: pd.Series,
    prob: np.ndarray,
    default_threshold: float,
    beta: float = 1.0,
) -> float:
    actual = safe_numeric(actual_density).fillna(0.0).clip(lower=0.0)
    y_true = (actual > 0.0).astype(int).to_numpy(dtype=int)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float(default_threshold)
    candidates = sorted({float(default_threshold), *[round(float(x), 4) for x in np.linspace(0.2, 0.8, 61)]})
    best_threshold = float(default_threshold)
    best_score = -1.0
    for threshold in candidates:
        y_pred = (prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        if precision <= 0.0 and recall <= 0.0:
            score = 0.0
        else:
            beta_sq = float(beta) ** 2
            score = (1 + beta_sq) * precision * recall / max(beta_sq * precision + recall, 1e-12)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def signature_from_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    prefix: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column in feature_columns:
        values = safe_numeric(df[column]) if column in df.columns else pd.Series(np.nan, index=df.index)
        out[f"{prefix}{column}__mean"] = float(values.mean()) if values.notna().any() else np.nan
        out[f"{prefix}{column}__std"] = float(values.std(ddof=0)) if values.notna().any() else np.nan
        out[f"{prefix}{column}__nonnull_ratio"] = float(values.notna().mean()) if len(values) else np.nan
    return out


def signature_value_columns(signature_df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in signature_df.columns
        if col.endswith("__mean") or col.endswith("__std")
    ]


def match_expert_for_frame(
    target_df: pd.DataFrame,
    strata_name: str,
    feature_columns: list[str],
    expert_signature_df: pd.DataFrame,
    expert_quality_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    matches = match_experts_for_frame(
        target_df=target_df,
        strata_name=strata_name,
        feature_columns=feature_columns,
        expert_signature_df=expert_signature_df,
        expert_quality_by_id=expert_quality_by_id,
        top_k=1,
    )
    if matches:
        return matches[0]
    return {
        "ExpertID": "",
        "ExpertWellSegment": "",
        "MatchDistance": np.nan,
        "MatchScore": 0.0,
        "QualityScore": 0.0,
        "WeightedMatchScore": 0.0,
        "MatchStatus": "missing_valid_expert_signature",
        "FallbackReason": "missing_valid_expert_signature",
    }


def match_experts_for_frame(
    target_df: pd.DataFrame,
    strata_name: str,
    feature_columns: list[str],
    expert_signature_df: pd.DataFrame,
    expert_quality_by_id: dict[str, dict[str, Any]] | None = None,
    top_k: int = 1,
) -> list[dict[str, Any]]:
    candidates = expert_signature_df[expert_signature_df["StrataName"] == strata_name].copy()
    if candidates.empty:
        return [{
            "ExpertID": "",
            "ExpertWellSegment": "",
            "MatchDistance": np.nan,
            "MatchScore": 0.0,
            "QualityScore": 0.0,
            "WeightedMatchScore": 0.0,
            "MatchStatus": "missing_strata_expert",
            "FallbackReason": "missing_strata_expert",
        }]
    target_sig = signature_from_frame(target_df, feature_columns)
    value_cols = signature_value_columns(candidates)
    out: list[dict[str, Any]] = []
    quality_lookup = expert_quality_by_id or {}
    for _, row in candidates.iterrows():
        diffs: list[float] = []
        for col in value_cols:
            target_value = target_sig.get(col, np.nan)
            expert_value = row.get(col, np.nan)
            if pd.isna(target_value) or pd.isna(expert_value):
                continue
            scale = float(np.nanstd(candidates[col].to_numpy(dtype=float)))
            if not np.isfinite(scale) or scale <= 1e-9:
                scale = max(abs(float(expert_value)), 1.0)
            diffs.append((float(target_value) - float(expert_value)) / scale)
        if diffs:
            distance = float(np.sqrt(np.mean(np.square(diffs))))
            status = "matched"
            reason = ""
        else:
            distance = float("inf")
            status = "matched_without_signature_overlap"
            reason = "no_numeric_signature_overlap"
        score = float(1.0 / (1.0 + distance)) if np.isfinite(distance) else 0.0
        expert_id = str(row["ExpertID"])
        quality = float(quality_lookup.get(expert_id, {}).get("ExpertQualityScore", 0.5))
        quality = float(np.clip(quality, 0.0, 1.0))
        weighted_score = score * (0.25 + 0.75 * quality)
        candidate = {
            "ExpertID": expert_id,
            "ExpertWellSegment": row["ExpertWellSegment"],
            "MatchDistance": distance if np.isfinite(distance) else np.nan,
            "MatchScore": score,
            "QualityScore": quality,
            "WeightedMatchScore": weighted_score,
            "MatchStatus": status,
            "FallbackReason": reason,
        }
        out.append(candidate)
    out.sort(key=lambda item: float(item.get("WeightedMatchScore", 0.0)), reverse=True)
    return out[: max(1, int(top_k))]


def predict_with_expert(
    df: pd.DataFrame,
    model_pack: dict[str, Any],
    gate_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feature_columns = list(model_pack.get("feature_columns") or [])
    if not feature_columns:
        raise RuntimeError(f"Expert pack missing feature_columns: {model_pack.get('expert_id', '')}")
    x = ensure_columns(df, feature_columns)[feature_columns].apply(pd.to_numeric, errors="coerce")
    stage1 = model_pack["stage1"]
    proba = stage1.predict_proba(x)
    if proba.shape[1] == 1:
        cls_value = int(stage1.named_steps["model"].classes_[0])
        prob = np.ones(len(df), dtype=float) if cls_value == 1 else np.zeros(len(df), dtype=float)
    else:
        classes = list(stage1.named_steps["model"].classes_)
        pos_index = classes.index(1) if 1 in classes else proba.shape[1] - 1
        prob = proba[:, pos_index]
    density_raw = np.maximum(model_pack["stage2"].predict(x), 0.0)
    if gate_threshold is None:
        density = density_raw
    else:
        density = density_raw * (prob >= float(gate_threshold)).astype(float)
    return prob, density


def predict_with_expert_raw(df: pd.DataFrame, model_pack: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return predict_with_expert(df, model_pack, gate_threshold=None)


def gate_density_by_threshold(prob: np.ndarray, density_raw: np.ndarray, threshold: float) -> np.ndarray:
    return np.maximum(density_raw, 0.0) * (prob >= float(threshold)).astype(float)


def expert_threshold(
    bundle: ExpertLibraryBundle,
    model_pack: dict[str, Any] | None,
    strata_name: str,
) -> float:
    expert_id = str((model_pack or {}).get("expert_id") or "")
    if expert_id and expert_id in bundle.stage1_thresholds_by_expert:
        return float(bundle.stage1_thresholds_by_expert[expert_id])
    if model_pack and model_pack.get("stage1_threshold") is not None:
        return float(model_pack["stage1_threshold"])
    return float(bundle.stage1_thresholds_by_strata.get(strata_name, bundle.stage1_threshold))


def quality_score_from_metrics(metrics: dict[str, Any], val_row_count: int) -> dict[str, Any]:
    f1 = float(metrics.get("F1", 0.0) or 0.0)
    recall = float(metrics.get("Recall", 0.0) or 0.0)
    precision = float(metrics.get("Precision", 0.0) or 0.0)
    r2 = float(metrics.get("DensityR2", np.nan))
    if not np.isfinite(r2):
        r2_component = 0.0
    else:
        r2_component = float(np.clip((r2 + 0.25) / 1.25, 0.0, 1.0))
    sample_component = float(np.clip(np.log10(max(int(val_row_count), 1)) / 4.0, 0.0, 1.0))
    balance_component = float(np.sqrt(max(precision, 0.0) * max(recall, 0.0))) if precision > 0 and recall > 0 else 0.0
    score = 0.60 * f1 + 0.20 * balance_component + 0.15 * r2_component + 0.05 * sample_component
    score = float(np.clip(score, 0.0, 1.0))
    if f1 < 0.15 or (f1 < 0.25 and r2 < -1.0):
        label = "weak"
    elif score < 0.35:
        label = "limited"
    else:
        label = "usable"
    return {
        "ExpertQualityScore": score,
        "ExpertQualityLabel": label,
        "F1Component": f1,
        "R2Component": r2_component,
        "SampleComponent": sample_component,
        "BalanceComponent": balance_component,
    }


def get_ensemble_top_k(config: dict[str, Any]) -> int:
    return max(1, int(config.get("expert_ensemble_top_k", 3)))


def threshold_caps(config: dict[str, Any]) -> dict[str, float]:
    defaults = {"沙三段": 0.30, "沙四段": 0.25}
    user_caps = config.get("stage1_threshold_caps_by_strata") or {}
    return {strata: float(user_caps.get(strata, defaults[strata])) for strata in TARGET_STRATA}


def adaptive_min_positive_ratios(config: dict[str, Any]) -> dict[str, float]:
    defaults = {"沙三段": 0.20, "沙四段": 0.15}
    user_values = config.get("adaptive_min_positive_ratio_by_strata") or {}
    return {strata: float(user_values.get(strata, defaults[strata])) for strata in TARGET_STRATA}


def resolve_prediction_threshold(
    config: dict[str, Any],
    bundle: ExpertLibraryBundle,
    strata_name: str,
    prob: np.ndarray,
    matches: list[dict[str, Any]],
) -> tuple[float, str, float]:
    strategy = str(config.get("threshold_strategy", "capped_expert_adaptive"))
    caps = threshold_caps(config)
    min_ratios = adaptive_min_positive_ratios(config)
    weighted: list[tuple[float, float]] = []
    for match in matches:
        expert_id = str(match.get("ExpertID", ""))
        if not expert_id:
            continue
        threshold = float(bundle.stage1_thresholds_by_expert.get(expert_id, bundle.stage1_thresholds_by_strata.get(strata_name, bundle.stage1_threshold)))
        weight = float(match.get("WeightedMatchScore", match.get("MatchScore", 0.0)) or 0.0)
        if weight > 0.0:
            weighted.append((threshold, weight))
    if weighted:
        base_threshold = float(sum(t * w for t, w in weighted) / max(sum(w for _, w in weighted), 1e-12))
    else:
        base_threshold = float(bundle.stage1_thresholds_by_strata.get(strata_name, bundle.stage1_threshold))
    capped_threshold = min(base_threshold, caps.get(strata_name, base_threshold))
    adaptive_threshold = capped_threshold
    if "adaptive" in strategy and len(prob):
        min_ratio = float(np.clip(min_ratios.get(strata_name, 0.0), 0.0, 0.95))
        if min_ratio > 0.0:
            adaptive_candidate = safe_quantile(pd.Series(prob), max(0.0, 1.0 - min_ratio))
            if np.isfinite(adaptive_candidate):
                adaptive_threshold = min(capped_threshold, adaptive_candidate)
    final_threshold = float(np.clip(adaptive_threshold, 0.05, 0.8))
    return final_threshold, strategy, base_threshold


def predict_with_expert_ensemble(
    df: pd.DataFrame,
    strata_name: str,
    experts_by_strata: dict[str, dict[str, dict[str, Any]]],
    matches: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    parts: list[tuple[np.ndarray, np.ndarray, float, str, str]] = []
    for match in matches:
        expert_id = str(match.get("ExpertID", ""))
        model_pack = experts_by_strata.get(strata_name, {}).get(expert_id)
        if not model_pack:
            continue
        prob, density_raw = predict_with_expert_raw(df, model_pack)
        weight = float(match.get("WeightedMatchScore", match.get("MatchScore", 0.0)) or 0.0)
        if weight <= 0.0:
            weight = 1e-6
        parts.append((prob, density_raw, weight, expert_id, str(match.get("ExpertWellSegment", ""))))
    if not parts:
        raise RuntimeError(f"No usable experts for strata {strata_name}")
    total_weight = max(sum(item[2] for item in parts), 1e-12)
    prob = sum(item[0] * item[2] for item in parts) / total_weight
    density_raw = sum(item[1] * item[2] for item in parts) / total_weight
    info = {
        "ExpertID": "+".join(item[3] for item in parts),
        "ExpertWellSegment": "+".join(item[4] for item in parts),
        "EnsembleExpertCount": int(len(parts)),
        "EnsembleWeightSum": float(total_weight),
        "MatchScore": float(np.mean([float(m.get("MatchScore", 0.0) or 0.0) for m in matches[: len(parts)]])),
        "WeightedMatchScore": float(np.mean([float(m.get("WeightedMatchScore", 0.0) or 0.0) for m in matches[: len(parts)]])),
        "QualityScore": float(np.mean([float(m.get("QualityScore", 0.0) or 0.0) for m in matches[: len(parts)]])),
    }
    return prob, density_raw, info


def evaluate_predictions(actual_density: pd.Series, pred_density: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    actual = safe_numeric(actual_density).fillna(0.0).clip(lower=0.0)
    actual_flag = (actual > 0.0).astype(int)
    pred_flag = (prob >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(actual_flag, pred_flag)),
        "Precision": float(precision_score(actual_flag, pred_flag, zero_division=0)),
        "Recall": float(recall_score(actual_flag, pred_flag, zero_division=0)),
        "F1": float(f1_score(actual_flag, pred_flag, zero_division=0)),
        "DensityMAE": float(mean_absolute_error(actual, pred_density)),
        "DensityRMSE": float(np.sqrt(mean_squared_error(actual, pred_density))),
        "DensityR2": float(r2_score(actual, pred_density)) if len(actual) >= 2 else np.nan,
        "ActualPositiveRatio": float(actual_flag.mean()) if len(actual_flag) else np.nan,
        "PredPositiveRatio": float(pred_flag.mean()) if len(pred_flag) else np.nan,
    }


def load_step2_interval(interval_csv: Path) -> pd.Series:
    df = pd.read_csv(interval_csv)
    if df.empty:
        raise ValueError(f"Empty interval table: {interval_csv}")
    return df.iloc[0]


def assign_step2_strata(main_df: pd.DataFrame, interval_row: pd.Series) -> pd.DataFrame:
    out = main_df.copy()
    depth = safe_numeric(out["DEPT"])
    t4 = float(interval_row["T4_DEPT"])
    t6 = float(interval_row["T6_DEPT"])
    t7 = float(interval_row["T7_DEPT"])
    out["StrataName"] = pd.NA
    out.loc[(depth >= t4) & (depth < t6), "StrataName"] = "沙三段"
    out.loc[(depth >= t6) & (depth <= t7), "StrataName"] = "沙四段"
    return out


def build_point_count_calibrators(
    config: dict[str, Any],
    bundle_feature_columns: list[str],
    experts_by_strata: dict[str, dict[str, dict[str, Any]]],
    train_df: pd.DataFrame,
    expert_signature_df: pd.DataFrame,
    holdout_well: str,
    stage1_thresholds_by_strata: dict[str, float],
    stage1_thresholds_by_expert: dict[str, float],
    expert_quality_by_id: dict[str, dict[str, Any]],
    deploy_thresholds_by_strata: dict[str, float],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    calibration_rows: list[dict[str, Any]] = []
    point_min_density = float(config.get("point_refine_min_density", 0.0))
    for (strata_name, well_segment), target_df in train_df[train_df["WellSegment"] != holdout_well].groupby(
        ["StrataName", "WellSegment"],
        dropna=False,
    ):
        strata_name = str(strata_name)
        well_segment = str(well_segment)
        matches = match_experts_for_frame(
            target_df,
            strata_name,
            bundle_feature_columns,
            expert_signature_df,
            expert_quality_by_id=expert_quality_by_id,
            top_k=get_ensemble_top_k(config),
        )
        if not matches or not matches[0].get("ExpertID"):
            continue
        prob, pred_density_raw, ensemble_info = predict_with_expert_ensemble(
            target_df,
            strata_name,
            experts_by_strata,
            matches,
        )
        threshold_bundle = ExpertLibraryBundle(
            feature_columns=bundle_feature_columns,
            signature_columns=bundle_feature_columns,
            min_nonnull_features=int(config.get("min_nonnull_features", 2)),
            target_column=str(config.get("target_column", "Density")),
            stage1_threshold=float(config.get("stage1_threshold", 0.5)),
            stage1_thresholds_by_strata=deploy_thresholds_by_strata,
            stage1_thresholds_by_expert=stage1_thresholds_by_expert,
            expert_quality_by_id=expert_quality_by_id,
            point_count_calibrators_by_strata={},
            experts_by_strata=experts_by_strata,
            metadata={},
        )
        strata_threshold, threshold_scope, _ = resolve_prediction_threshold(config, threshold_bundle, strata_name, prob, matches)
        pred_density = gate_density_by_threshold(prob, pred_density_raw, strata_threshold)
        pred_df = target_df[
            ["SampleID", "WellSegment", "X", "Y", "TIME", "TVD", "DEPT", "StrataName", "GT_POINT_FLAG"]
        ].copy()
        pred_df["WellName"] = pred_df["WellSegment"]
        pred_df["PredFractureProb"] = prob
        pred_df["PredDensityRaw"] = pred_density_raw
        pred_df["PredDensity"] = pred_density
        pred_df["PredHasFracture"] = (pred_df["PredFractureProb"] >= strata_threshold).astype(int)
        pred_df["ExpertID"] = ensemble_info["ExpertID"]
        pred_df["ExpertWellSegment"] = ensemble_info["ExpertWellSegment"]
        pred_df["MatchScore"] = ensemble_info["MatchScore"]
        pred_df["Stage1Threshold"] = strata_threshold
        work, payloads, depth_col = extract_segment_payloads(
            pred_df,
            threshold=strata_threshold,
            min_density=point_min_density,
            shape_mode="density_prob",
        )
        gt_depths = safe_numeric(
            work.loc[safe_numeric(work["GT_POINT_FLAG"]).fillna(0.0) > 0.0, depth_col]
        ).dropna().to_numpy(dtype=np.float64)
        for payload in payloads:
            calibration_rows.append(
                {
                    "StrataName": strata_name,
                    "WellSegment": well_segment,
                    "ExpertID": ensemble_info["ExpertID"],
                    "SegStartDepth": payload["SegStartDepth"],
                    "SegEndDepth": payload["SegEndDepth"],
                    "SegLength": payload["SegLength"],
                    "RawDensityTotal": payload["RawDensityTotal"],
                    "ProbMass": payload["ProbMass"],
                    "ProbMean": payload["ProbMean"],
                    "DensityMean": payload["DensityMean"],
                    "GTPointCountInPredSegment": count_points_in_depth_range(
                        gt_depths,
                        payload["SegStartDepth"],
                        payload["SegEndDepth"],
                    ),
                    "Stage1Threshold": strata_threshold,
                    "ThresholdScope": threshold_scope,
                }
            )

    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(output_dir / "point_count_calibration_segments.csv", index=False, encoding="utf-8-sig")
    calibrators: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for strata_name in TARGET_STRATA:
        strata_df = calibration_df[calibration_df["StrataName"] == strata_name].copy() if not calibration_df.empty else pd.DataFrame()
        calibrator = fit_point_count_calibrator(strata_df, basis_col="RawDensityTotal")
        calibrator["strata_name"] = strata_name
        calibrator["stage1_threshold"] = float(stage1_thresholds_by_strata.get(strata_name, config.get("stage1_threshold", 0.5)))
        calibrator["point_refine_min_density"] = point_min_density
        calibrators[strata_name] = calibrator
        summary_rows.append(
            {
                "StrataName": strata_name,
                "BasisCol": calibrator["basis_col"],
                "Scale": calibrator["scale"],
                "BasisSum": calibrator["basis_sum"],
                "TargetSum": calibrator["target_sum"],
                "SegmentCount": calibrator["segment_count"],
                "WellCount": calibrator["well_count"],
                "Stage1Threshold": calibrator["stage1_threshold"],
                "PointRefineMinDensity": point_min_density,
                "UsedWellCount": calibrator.get("used_well_count", 0),
                "DroppedZeroCoverWellCount": calibrator.get("dropped_zero_cover_well_count", 0),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "point_count_calibration_summary.csv", index=False, encoding="utf-8-sig")
    return calibrators


def resolve_deploy_thresholds_by_strata(
    train_df: pd.DataFrame,
    stage1_thresholds_by_strata: dict[str, float],
    target_column: str,
) -> dict[str, float]:
    deploy_thresholds: dict[str, float] = {}
    for strata_name in TARGET_STRATA:
        train_pos_ratio = float(
            safe_numeric(train_df.loc[train_df["StrataName"] == strata_name, target_column]).fillna(0.0).gt(0.0).mean()
        ) if not train_df.loc[train_df["StrataName"] == strata_name].empty else 0.0
        calibrated = float(stage1_thresholds_by_strata.get(strata_name, 0.5))
        if train_pos_ratio >= 0.30:
            deploy_thresholds[strata_name] = min(calibrated, 0.5)
        elif train_pos_ratio >= 0.20:
            deploy_thresholds[strata_name] = min(calibrated, 0.55)
        else:
            deploy_thresholds[strata_name] = min(calibrated, 0.6)
    return deploy_thresholds


def train_library(config: dict[str, Any]) -> ExpertLibraryBundle:
    groups_dir = Path(config.get("step3_groups_dir") or STEP3_GROUPS_DEFAULT)
    step2_root = Path(config.get("step2_output_root") or STEP2_ROOT_DEFAULT)
    output_dir = Path(config["output_dir"])
    library_dir = output_dir / "expert_library"
    ensure_dir(library_dir)

    holdout_well = str(config.get("holdout_well") or DEFAULT_HOLDOUT_WELL)
    min_nonnull_features = int(config.get("min_nonnull_features", 2))
    min_expert_samples = int(config.get("min_expert_samples", 50))
    target_column = str(config.get("target_column", "Density"))
    threshold = float(config.get("stage1_threshold", 0.5))
    random_state = int(config.get("random_state", 42))
    cls_estimators = int(config.get("stage1_n_estimators", 240))
    reg_estimators = int(config.get("stage2_n_estimators", 280))

    source_df, input_manifest = read_step3_groups(groups_dir)
    step2_wells = step2_ok_wells(step2_root)["WellName"].astype(str).tolist()
    target_probe = load_step2_main_for_feature_scan(step2_root, step2_wells)
    feature_columns, feature_contract = select_feature_columns(config, source_df, target_probe)
    train_df = usable_training_frame(source_df, feature_columns, min_nonnull_features, target_column)
    if train_df.empty:
        raise RuntimeError("No usable Step 3 group samples for Step 4 training.")

    input_manifest["SplitRole"] = np.where(input_manifest["WellSegment"] == holdout_well, "holdout_validation", "train_candidate")
    input_manifest.to_csv(output_dir / "input_group_manifest.csv", index=False, encoding="utf-8-sig")
    feature_contract.to_csv(output_dir / "feature_contract.csv", index=False, encoding="utf-8-sig")

    split_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    experts_by_strata: dict[str, dict[str, dict[str, Any]]] = {}
    stage1_thresholds_by_strata: dict[str, float] = {strata: threshold for strata in TARGET_STRATA}
    stage1_thresholds_by_expert: dict[str, float] = {}
    threshold_beta = float(config.get("stage1_threshold_beta", 1.0))

    grouped = train_df.groupby(["StrataName", "WellSegment"], dropna=False)
    for (strata_name, well_segment), group_df in grouped:
        strata_name = str(strata_name)
        well_segment = str(well_segment)
        split_role = "holdout_validation" if well_segment == holdout_well else "train"
        split_rows.append(
            {
                "GroupID": str(group_df["GroupID"].iloc[0]),
                "WellSegment": well_segment,
                "StrataName": strata_name,
                "SplitRole": split_role,
                "UsableRowCount": int(len(group_df)),
                "PositiveDensityRows": int((safe_numeric(group_df[target_column]).fillna(0.0) > 0.0).sum()),
                "LeakageCheck": "pass" if split_role != "train" or well_segment != holdout_well else "fail",
            }
        )
        if split_role != "train":
            continue
        if strata_name not in TARGET_STRATA or len(group_df) < min_expert_samples:
            continue
        expert_id = f"expert_{strata_name}_{well_segment}"
        train_subset = train_df[
            (train_df["StrataName"] == strata_name)
            & (train_df["WellSegment"] != well_segment)
            & (train_df["WellSegment"] != holdout_well)
        ].copy()
        if train_subset.empty:
            continue
        cls_model, reg_model, train_stats, model_feature_columns = train_one_expert(
            train_df=train_subset,
            feature_columns=feature_columns,
            target_column=target_column,
            random_state=random_state,
            cls_estimators=cls_estimators,
            reg_estimators=reg_estimators,
        )
        strata_dir = library_dir / safe_file_stem(strata_name)
        ensure_dir(strata_dir)
        stage1_path = strata_dir / f"{safe_file_stem(expert_id)}_stage1_presence_model.joblib"
        stage2_path = strata_dir / f"{safe_file_stem(expert_id)}_stage2_density_model.joblib"
        joblib.dump(cls_model, stage1_path)
        joblib.dump(reg_model, stage2_path)
        model_pack = {
            "stage1": cls_model,
            "stage2": reg_model,
            "expert_id": expert_id,
            "expert_well_segment": well_segment,
            "strata_name": strata_name,
            "feature_columns": model_feature_columns,
            "stage1_threshold": threshold,
        }
        experts_by_strata.setdefault(strata_name, {})[expert_id] = model_pack
        registry_rows.append(
            {
                "ExpertID": expert_id,
                "ExpertType": "well_level",
                "LibraryStrata": strata_name,
                "ExpertWellSegment": well_segment,
                "SourceGroupID": str(group_df["GroupID"].iloc[0]),
                "TrainSampleCount": int(len(train_subset)),
                "PositiveDensityRows": train_stats["PositiveDensityRows"],
                "MeanDensity": train_stats["MeanDensity"],
                "FeatureColumns": ",".join(model_feature_columns),
                "TargetColumn": target_column,
                "Stage1Threshold": threshold,
                "Stage1ModelPath": str(stage1_path),
                "Stage2ModelPath": str(stage2_path),
            }
        )
        signature_rows.append(
            {
                "ExpertID": expert_id,
                "StrataName": strata_name,
                "ExpertWellSegment": well_segment,
                "SourceGroupID": str(group_df["GroupID"].iloc[0]),
                "TrainSampleCount": int(len(train_subset)),
                **signature_from_frame(group_df, model_feature_columns),
            }
        )

    if not experts_by_strata:
        raise RuntimeError("No well-level experts trained. Check holdout/min_expert_samples/feature coverage.")

    registry_df = pd.DataFrame(registry_rows)
    signature_df = pd.DataFrame(signature_rows)
    split_df = pd.DataFrame(split_rows)
    signature_df.to_csv(library_dir / "expert_signature.csv", index=False, encoding="utf-8-sig")
    split_df.to_csv(output_dir / "train_holdout_split_manifest.csv", index=False, encoding="utf-8-sig")

    threshold_calibration_rows: list[dict[str, Any]] = []
    threshold_summary_rows: list[dict[str, Any]] = []
    for (strata_name, well_segment), val_df in train_df[train_df["WellSegment"] != holdout_well].groupby(
        ["StrataName", "WellSegment"],
        dropna=False,
    ):
        strata_name = str(strata_name)
        well_segment = str(well_segment)
        match = match_expert_for_frame(val_df, strata_name, feature_columns, signature_df)
        model_pack = experts_by_strata.get(strata_name, {}).get(match["ExpertID"])
        if not model_pack:
            continue
        prob, pred_density_raw = predict_with_expert_raw(val_df, model_pack)
        actual = safe_numeric(val_df[target_column]).fillna(0.0).clip(lower=0.0)
        calibrated_threshold = calibrate_stage1_threshold(actual, prob, threshold, beta=threshold_beta)
        stage1_thresholds_by_expert[str(match["ExpertID"])] = calibrated_threshold
        model_pack["stage1_threshold"] = calibrated_threshold
        pred_density = gate_density_by_threshold(prob, pred_density_raw, calibrated_threshold)
        metrics = evaluate_predictions(actual, pred_density, prob, calibrated_threshold)
        threshold_summary_rows.append(
            {
                "ExpertID": match["ExpertID"],
                "ExpertWellSegment": match["ExpertWellSegment"],
                "ValidationWellSegment": well_segment,
                "StrataName": strata_name,
                "Stage1Threshold": calibrated_threshold,
                "ThresholdScope": "expert",
                "ThresholdBeta": threshold_beta,
                "ValRowCount": int(len(val_df)),
                **metrics,
            }
        )
        threshold_calibration_rows.append(
            pd.DataFrame(
                {
                    "StrataName": strata_name,
                    "WellSegment": well_segment,
                    "SampleID": val_df["SampleID"].astype(str).to_numpy(),
                    "ActualDensity": actual.to_numpy(dtype=float),
                    "PredFractureProb": prob,
                    "PredDensityRaw": pred_density_raw,
                    "PredDensity": pred_density,
                    "Stage1Threshold": calibrated_threshold,
                    "ExpertID": match["ExpertID"],
                    "ExpertWellSegment": match["ExpertWellSegment"],
                }
            )
        )
    threshold_calibration_df = (
        pd.concat(threshold_calibration_rows, ignore_index=True) if threshold_calibration_rows else pd.DataFrame()
    )
    threshold_calibration_df.to_csv(output_dir / "stage1_threshold_calibration_loo_rows.csv", index=False, encoding="utf-8-sig")
    threshold_summary_df = pd.DataFrame(threshold_summary_rows)
    expert_quality_by_id: dict[str, dict[str, Any]] = {}
    if not threshold_summary_df.empty:
        quality_rows: list[dict[str, Any]] = []
        for idx, row in threshold_summary_df.iterrows():
            metrics = row.to_dict()
            quality = quality_score_from_metrics(metrics, int(row.get("ValRowCount", 0) or 0))
            expert_id = str(row["ExpertID"])
            expert_quality_by_id[expert_id] = {
                **quality,
                "ValidationF1": float(row.get("F1", 0.0) or 0.0),
                "ValidationDensityR2": float(row.get("DensityR2", np.nan)),
                "ValidationDensityMAE": float(row.get("DensityMAE", np.nan)),
                "ValidationRowCount": int(row.get("ValRowCount", 0) or 0),
            }
            for key, value in quality.items():
                threshold_summary_df.loc[idx, key] = value
            quality_rows.append({"ExpertID": expert_id, **row.to_dict(), **quality})
        pd.DataFrame(quality_rows).to_csv(output_dir / "expert_quality_report.csv", index=False, encoding="utf-8-sig")
        signature_df["ExpertQualityScore"] = signature_df["ExpertID"].astype(str).map(
            lambda expert_id: expert_quality_by_id.get(expert_id, {}).get("ExpertQualityScore", 0.5)
        )
        signature_df["ExpertQualityLabel"] = signature_df["ExpertID"].astype(str).map(
            lambda expert_id: expert_quality_by_id.get(expert_id, {}).get("ExpertQualityLabel", "unknown")
        )
        for strata_experts in experts_by_strata.values():
            for expert_id, model_pack in strata_experts.items():
                model_pack.update(expert_quality_by_id.get(expert_id, {}))
    threshold_summary_df.to_csv(output_dir / "stage1_threshold_by_expert.csv", index=False, encoding="utf-8-sig")
    if not threshold_summary_df.empty:
        for strata_name, strata_df in threshold_summary_df.groupby("StrataName"):
            stage1_thresholds_by_strata[str(strata_name)] = float(safe_numeric(strata_df["Stage1Threshold"]).median())

    deploy_thresholds_by_strata = dict(stage1_thresholds_by_strata)
    if not registry_df.empty:
        registry_df["Stage1Threshold"] = registry_df["ExpertID"].astype(str).map(stage1_thresholds_by_expert).fillna(threshold)
        registry_df["ThresholdScope"] = "expert"
        registry_df["ThresholdBeta"] = threshold_beta
        registry_df["ExpertQualityScore"] = registry_df["ExpertID"].astype(str).map(
            lambda expert_id: expert_quality_by_id.get(expert_id, {}).get("ExpertQualityScore", 0.5)
        )
        registry_df["ExpertQualityLabel"] = registry_df["ExpertID"].astype(str).map(
            lambda expert_id: expert_quality_by_id.get(expert_id, {}).get("ExpertQualityLabel", "unknown")
        )
    registry_df.to_csv(library_dir / "expert_registry.csv", index=False, encoding="utf-8-sig")
    signature_df.to_csv(library_dir / "expert_signature.csv", index=False, encoding="utf-8-sig")

    point_count_calibrators_by_strata = build_point_count_calibrators(
        config=config,
        bundle_feature_columns=feature_columns,
        experts_by_strata=experts_by_strata,
        train_df=train_df,
        expert_signature_df=signature_df,
        holdout_well=holdout_well,
        stage1_thresholds_by_strata=stage1_thresholds_by_strata,
        stage1_thresholds_by_expert=stage1_thresholds_by_expert,
        expert_quality_by_id=expert_quality_by_id,
        deploy_thresholds_by_strata=deploy_thresholds_by_strata,
        output_dir=output_dir,
    )

    bundle = ExpertLibraryBundle(
        feature_columns=feature_columns,
        signature_columns=feature_columns,
        min_nonnull_features=min_nonnull_features,
        target_column=target_column,
        stage1_threshold=threshold,
        stage1_thresholds_by_strata=deploy_thresholds_by_strata,
        stage1_thresholds_by_expert=stage1_thresholds_by_expert,
        expert_quality_by_id=expert_quality_by_id,
        point_count_calibrators_by_strata=point_count_calibrators_by_strata,
        experts_by_strata=experts_by_strata,
        metadata={
            "step3_groups_dir": str(groups_dir),
            "step2_output_root": str(step2_root),
            "holdout_well": holdout_well,
            "group_count": int(len(input_manifest)),
            "training_group_count": int((split_df["SplitRole"] == "train").sum()),
            "holdout_group_count": int((split_df["SplitRole"] == "holdout_validation").sum()),
            "expert_count": int(len(registry_df)),
            "expert_design": "strata_first_well_level_expert_library",
            "density_role": "main_supervision",
            "point_role": "auxiliary_validation_and_postprocess_only",
        },
    )
    joblib.dump(bundle, library_dir / "expert_library_bundle.joblib")
    write_json(
        library_dir / "expert_library_summary.json",
        {
            **bundle.metadata,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "stage1_threshold": threshold,
            "stage1_thresholds_by_strata": deploy_thresholds_by_strata,
            "stage1_thresholds_by_expert": stage1_thresholds_by_expert,
            "expert_quality_by_id": expert_quality_by_id,
            "stage1_threshold_beta": threshold_beta,
            "threshold_scope": str(config.get("threshold_strategy", "capped_expert_adaptive")),
            "expert_ensemble_top_k": get_ensemble_top_k(config),
            "stage1_threshold_caps_by_strata": threshold_caps(config),
            "adaptive_min_positive_ratio_by_strata": adaptive_min_positive_ratios(config),
            "point_count_calibrators_by_strata": point_count_calibrators_by_strata,
            "experts_by_strata": {k: sorted(v.keys()) for k, v in experts_by_strata.items()},
        },
    )
    run_holdout_validation(config, bundle, train_df, signature_df, output_dir)
    return bundle


def run_holdout_validation(
    config: dict[str, Any],
    bundle: ExpertLibraryBundle,
    train_df: pd.DataFrame,
    expert_signature_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    holdout_well = str(config.get("holdout_well") or DEFAULT_HOLDOUT_WELL)
    validation_dir = output_dir / "validation"
    ensure_dir(validation_dir)
    holdout_df = train_df[train_df["WellSegment"] == holdout_well].copy()
    match_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    point_metric_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    if holdout_df.empty:
        pd.DataFrame().to_csv(validation_dir / "holdout_expert_match_manifest.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(validation_dir / "holdout_metrics_by_strata.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(validation_dir / "holdout_point_metrics_by_strata.csv", index=False, encoding="utf-8-sig")
        return
    for strata_name, val_df in holdout_df.groupby("StrataName"):
        strata_name = str(strata_name)
        point_calibrator = dict(bundle.point_count_calibrators_by_strata.get(strata_name, {}))
        matches = match_experts_for_frame(
            val_df,
            strata_name,
            bundle.feature_columns,
            expert_signature_df,
            expert_quality_by_id=bundle.expert_quality_by_id,
            top_k=get_ensemble_top_k(config),
        )
        match = matches[0] if matches else {}
        model_pack = bundle.experts_by_strata.get(strata_name, {}).get(str(match.get("ExpertID", "")))
        if model_pack:
            prob, pred_density_raw, ensemble_info = predict_with_expert_ensemble(
                val_df,
                strata_name,
                bundle.experts_by_strata,
                matches,
            )
            strata_threshold, threshold_scope, base_threshold = resolve_prediction_threshold(config, bundle, strata_name, prob, matches)
        else:
            prob = np.array([], dtype=float)
            pred_density_raw = np.array([], dtype=float)
            ensemble_info = {"ExpertID": "", "ExpertWellSegment": "", "EnsembleExpertCount": 0, "MatchScore": 0.0, "WeightedMatchScore": 0.0, "QualityScore": 0.0}
            strata_threshold = expert_threshold(bundle, model_pack, strata_name)
            threshold_scope = "strata_fallback"
            base_threshold = strata_threshold
        row = {
            "HoldoutWell": holdout_well,
            "StrataName": strata_name,
            "ValRowCount": int(len(val_df)),
            "Stage1Threshold": strata_threshold,
            "BaseStage1Threshold": base_threshold,
            "ThresholdScope": threshold_scope if model_pack else "strata_fallback",
            "EnsembleExpertCount": ensemble_info["EnsembleExpertCount"],
            "EnsembleExpertID": ensemble_info["ExpertID"],
            "EnsembleExpertWellSegment": ensemble_info["ExpertWellSegment"],
            "EnsembleQualityScore": ensemble_info["QualityScore"],
            **match,
        }
        match_rows.append(row)
        if not model_pack:
            continue
        pred_density = gate_density_by_threshold(prob, pred_density_raw, strata_threshold)
        sensitivity_thresholds = sorted(
            {
                float(strata_threshold),
                float(base_threshold),
                float(bundle.stage1_thresholds_by_strata.get(strata_name, bundle.stage1_threshold)),
                float(threshold_caps(config).get(strata_name, strata_threshold)),
                0.2,
                0.3,
                0.32,
                0.38,
                0.44,
                0.5,
            }
        )
        for test_threshold in sensitivity_thresholds:
            test_pred_density = gate_density_by_threshold(prob, pred_density_raw, test_threshold)
            sensitivity_rows.append(
                {
                    "HoldoutWell": holdout_well,
                    "StrataName": strata_name,
                    "TestThreshold": test_threshold,
                    "SelectedAsFinalThreshold": bool(abs(test_threshold - strata_threshold) < 1e-9),
                    "ExpertID": ensemble_info["ExpertID"],
                    "ExpertWellSegment": ensemble_info["ExpertWellSegment"],
                    **evaluate_predictions(val_df[bundle.target_column], test_pred_density, prob, test_threshold),
                }
            )
        pred_out = val_df[
            [
                "SampleID",
                "WellSegment",
                "X",
                "Y",
                "TIME",
                "TVD",
                "DEPT",
                "StrataName",
                "HasFracture",
                "GT_POINT_FLAG",
                *bundle.feature_columns,
                bundle.target_column,
            ]
        ].copy()
        pred_out = pred_out.rename(columns={"WellSegment": "WellName", bundle.target_column: "ActualDensity", "HasFracture": "ActualHasFracture"})
        pred_out["PredFractureProb"] = prob
        pred_out["PredDensityRaw"] = pred_density_raw
        pred_out["PredDensity"] = pred_density
        pred_out["PredHasFracture"] = (pred_out["PredFractureProb"] >= strata_threshold).astype(int)
        pred_out["ExpertID"] = ensemble_info["ExpertID"]
        pred_out["ExpertWellSegment"] = ensemble_info["ExpertWellSegment"]
        pred_out["MatchDistance"] = match.get("MatchDistance", np.nan)
        pred_out["MatchScore"] = ensemble_info["MatchScore"]
        pred_out["WeightedMatchScore"] = ensemble_info["WeightedMatchScore"]
        pred_out["ExpertQualityScore"] = ensemble_info["QualityScore"]
        pred_out["EnsembleExpertCount"] = ensemble_info["EnsembleExpertCount"]
        pred_out["PredictionStatus"] = "predicted"
        pred_out["Stage1Threshold"] = strata_threshold
        pred_out["BaseStage1Threshold"] = base_threshold
        pred_out["ThresholdScope"] = threshold_scope
        pred_out.to_csv(
            validation_dir / f"holdout_predictions_{safe_file_stem(holdout_well)}_{safe_file_stem(strata_name)}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        holdout_points = refine_points(
            pred_out,
            threshold=strata_threshold,
            min_density=float(config.get("point_refine_min_density", 0.0)),
            calibrator=point_calibrator,
        )
        holdout_points.to_csv(
            validation_dir / f"holdout_refined_points_{safe_file_stem(holdout_well)}_{safe_file_stem(strata_name)}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        depth_col = "DEPT" if "DEPT" in pred_out.columns else "TIME"
        gt_depths = safe_numeric(
            pred_out.loc[safe_numeric(pred_out["GT_POINT_FLAG"]).fillna(0.0) > 0.0, depth_col]
        ).dropna().to_numpy(dtype=np.float64)
        pred_depths = safe_numeric(holdout_points.get(depth_col, pd.Series(dtype=float))).dropna().to_numpy(dtype=np.float64)
        pred_to_gt = nearest_distance_stats(pred_depths, gt_depths)
        gt_to_pred = nearest_distance_stats(gt_depths, pred_depths)
        point_metric_rows.append(
            {
                "HoldoutWell": holdout_well,
                "StrataName": strata_name,
                "ExpertID": ensemble_info["ExpertID"],
                "ExpertWellSegment": ensemble_info["ExpertWellSegment"],
                "Stage1Threshold": strata_threshold,
                "PointCountBasisCol": point_calibrator.get("basis_col", "RawDensityTotal"),
                "PointCountScale": point_calibrator.get("scale", 1.0),
                "GTPointCount": int(len(gt_depths)),
                "PredPointCount": int(len(pred_depths)),
                "CountDiff": int(len(pred_depths) - len(gt_depths)),
                "PredToGTMeanDist": pred_to_gt["mean"],
                "PredToGTP90Dist": pred_to_gt["p90"],
                "GTToPredMeanDist": gt_to_pred["mean"],
                "GTToPredP90Dist": gt_to_pred["p90"],
            }
        )
        metric_rows.append(
            {
                "HoldoutWell": holdout_well,
                "StrataName": strata_name,
                "ExpertID": ensemble_info["ExpertID"],
                "ExpertWellSegment": ensemble_info["ExpertWellSegment"],
                "ValRowCount": int(len(val_df)),
                "Stage1Threshold": strata_threshold,
                "BaseStage1Threshold": base_threshold,
                "ThresholdScope": threshold_scope,
                "EnsembleExpertCount": ensemble_info["EnsembleExpertCount"],
                "EnsembleQualityScore": ensemble_info["QualityScore"],
                **evaluate_predictions(val_df[bundle.target_column], pred_density, prob, strata_threshold),
            }
        )
    pd.DataFrame(match_rows).to_csv(validation_dir / "holdout_expert_match_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(validation_dir / "holdout_metrics_by_strata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(point_metric_rows).to_csv(validation_dir / "holdout_point_metrics_by_strata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity_rows).to_csv(validation_dir / "holdout_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")


def refine_points(
    pred_df: pd.DataFrame,
    threshold: float,
    min_density: float,
    calibrator: dict[str, Any] | None = None,
) -> pd.DataFrame:
    point_columns = [
        "SourceSampleID",
        "WellName",
        "X",
        "Y",
        "TIME",
        "TVD",
        "DEPT",
        "StrataName",
        "Density",
        "HasFracture",
        "PredFractureProb",
        "PredDensityRaw",
        "PredDensity",
        "PredHasFracture",
        "Stage1Threshold",
        "SelectedAsFracturePoint",
        "ExpertID",
        "ExpertWellSegment",
        "MatchScore",
        "FractureRunID",
        "RunLength",
        "RunStartSampleID",
        "RunEndSampleID",
        "CountBasisCol",
        "CountBasisValue",
        "PointCountScale",
        "CalibratedCountValue",
        "PointSelectionRule",
    ]

    work, payloads, depth_col = extract_segment_payloads(
        pred_df,
        threshold=threshold,
        min_density=min_density,
        shape_mode=str((calibrator or {}).get("shape_mode", "density_prob")),
    )
    if not payloads:
        return pd.DataFrame(columns=point_columns)

    rows: list[dict[str, Any]] = []
    calibrator = calibrator or {"basis_col": "RawDensityTotal", "scale": 1.0, "mode": "global_density_scale"}
    basis_col = str(calibrator.get("basis_col", "RawDensityTotal"))
    scale = float(calibrator.get("scale", 1.0))
    for payload in payloads:
        basis_value = float(payload.get(basis_col, 0.0) or 0.0)
        count_value = max(basis_value * scale, 0.0)
        n_points = compute_n_points(count_value, min_points=1, rounding_mode="round")
        if n_points <= 0:
            continue
        picked_depths, picked_idx = pick_points_by_equal_intensity(
            payload["depth_seg"],
            payload["shape_intensity"],
            n_points,
        )
        for point_id, rel_idx in enumerate(picked_idx, start=1):
            abs_idx = int(payload["StartIdx"] + int(rel_idx))
            selected = work.iloc[abs_idx]
            rows.append(
                {
                    "SourceSampleID": selected.get("SampleID"),
                    "WellName": selected.get("WellName"),
                    "X": selected.get("X"),
                    "Y": selected.get("Y"),
                    "TIME": selected.get("TIME"),
                    "TVD": selected.get("TVD"),
                    "DEPT": selected.get("DEPT"),
                    "StrataName": selected.get("StrataName"),
                    "Density": float(selected.get("PredDensity", np.nan)),
                    "HasFracture": int(selected.get("PredHasFracture", 0)),
                    "PredFractureProb": float(selected.get("PredFractureProb", np.nan)),
                    "PredDensityRaw": float(selected.get("PredDensityRaw", np.nan)),
                    "PredDensity": float(selected.get("PredDensity", np.nan)),
                    "PredHasFracture": int(selected.get("PredHasFracture", 0)),
                    "Stage1Threshold": float(selected.get("Stage1Threshold", threshold)),
                    "SelectedAsFracturePoint": 1,
                    "ExpertID": selected.get("ExpertID"),
                    "ExpertWellSegment": selected.get("ExpertWellSegment"),
                    "MatchScore": selected.get("MatchScore"),
                    "FractureRunID": f"run_{int(payload['Segment_ID'])}",
                    "RunLength": int(payload["EndIdx"] - payload["StartIdx"] + 1),
                    "RunStartSampleID": str(work.iloc[int(payload["StartIdx"])].get("SampleID", "")),
                    "RunEndSampleID": str(work.iloc[int(payload["EndIdx"])].get("SampleID", "")),
                    "CountBasisCol": basis_col,
                    "CountBasisValue": basis_value,
                    "PointCountScale": scale,
                    "CalibratedCountValue": count_value,
                    "PointSelectionRule": (
                        f"PredFractureProb>={threshold};PredDensity>={min_density};"
                        f"{calibrator.get('mode', 'global_density_scale')};shape={calibrator.get('shape_mode', 'density_prob')}"
                    ),
                }
            )
    return pd.DataFrame(rows, columns=point_columns)


def load_expert_signature(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "expert_library/expert_signature.csv"
    if not path.exists():
        raise RuntimeError(f"Missing expert signature table: {path}")
    return pd.read_csv(path)


def predict_all_wells(config: dict[str, Any], bundle: ExpertLibraryBundle | None = None) -> None:
    output_dir = Path(config["output_dir"])
    ensure_dir(output_dir)
    if bundle is None:
        bundle_path = Path(config.get("model_bundle") or output_dir / "expert_library/expert_library_bundle.joblib")
        bundle = joblib.load(bundle_path)
    expert_signature_df = load_expert_signature(output_dir)

    step2_root = Path(config.get("step2_output_root") or STEP2_ROOT_DEFAULT)
    step2_summary = read_step2_summary(step2_root)
    ok_targets = step2_summary[step2_summary["Status"].astype(str) == "ok"].copy()
    if config.get("selected_wells"):
        selected = set(str(well) for well in config["selected_wells"])
        ok_targets = ok_targets[ok_targets["WellName"].astype(str).isin(selected)].copy()
    skipped_targets = step2_summary[step2_summary["Status"].astype(str) != "ok"].copy()
    skipped_targets.to_csv(output_dir / "skipped_step2_wells_manifest.csv", index=False, encoding="utf-8-sig")
    well_output_root = output_dir / "real_well_predictions"
    ensure_dir(well_output_root)

    point_threshold = config.get("point_refine_prob_threshold")
    point_min_density = float(config.get("point_refine_min_density", 0.0))
    summary_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    point_parts: list[pd.DataFrame] = []

    for _, target_row in ok_targets.iterrows():
        well_name = str(target_row["WellName"])
        main_csv = Path(str(target_row["MainCsv"]))
        interval_csv = Path(str(target_row["IntervalCsv"]))
        if not main_csv.exists() or not interval_csv.exists():
            summary_rows.append({"WellName": well_name, "Status": "missing_step2_inputs", "TotalRowCount": 0})
            continue

        main_df = pd.read_csv(main_csv)
        main_df = assign_step2_strata(main_df, load_step2_interval(interval_csv))
        main_df["PredFractureProb"] = np.nan
        main_df["PredDensityRaw"] = np.nan
        main_df["PredDensity"] = np.nan
        main_df["PredHasFracture"] = 0
        main_df["Stage1Threshold"] = np.nan
        main_df["BaseStage1Threshold"] = np.nan
        main_df["ThresholdScope"] = pd.NA
        main_df["PredictionStatus"] = "not_predicted"
        main_df["ExpertID"] = pd.NA
        main_df["ExpertWellSegment"] = pd.NA
        main_df["MatchDistance"] = np.nan
        main_df["MatchScore"] = np.nan
        main_df["WeightedMatchScore"] = np.nan
        main_df["ExpertQualityScore"] = np.nan
        main_df["EnsembleExpertCount"] = 0
        main_df["FallbackReason"] = pd.NA

        usable_mask = (
            safe_bool(main_df.get("SampleUsableForModel", pd.Series(True, index=main_df.index)), default=True)
            .reindex(main_df.index, fill_value=True)
            & build_feature_mask(main_df, bundle.feature_columns, bundle.min_nonnull_features)
        )

        for strata_name in TARGET_STRATA:
            strata_mask = main_df["StrataName"] == strata_name
            target_strata_df = main_df[strata_mask].copy()
            if target_strata_df.empty:
                match_rows.append(
                    {
                        "WellName": well_name,
                        "StrataName": strata_name,
                        "TargetRowCount": 0,
                        "UsableRowCount": 0,
                        "ExpertID": "",
                        "ExpertWellSegment": "",
                        "MatchStatus": "no_target_rows",
                        "FallbackReason": "no_target_rows",
                    }
                )
                continue
            usable_target_df = target_strata_df[usable_mask[strata_mask]]
            matches = match_experts_for_frame(
                usable_target_df,
                strata_name,
                bundle.feature_columns,
                expert_signature_df,
                expert_quality_by_id=bundle.expert_quality_by_id,
                top_k=get_ensemble_top_k(config),
            )
            match = matches[0] if matches else {}
            model_pack = bundle.experts_by_strata.get(strata_name, {}).get(str(match.get("ExpertID", "")))
            predict_mask = strata_mask & usable_mask
            if model_pack and predict_mask.any():
                prob, density_raw, ensemble_info = predict_with_expert_ensemble(
                    main_df.loc[predict_mask],
                    strata_name,
                    bundle.experts_by_strata,
                    matches,
                )
                strata_threshold, threshold_scope, base_threshold = resolve_prediction_threshold(config, bundle, strata_name, prob, matches)
            else:
                ensemble_info = {"ExpertID": "", "ExpertWellSegment": "", "EnsembleExpertCount": 0, "MatchScore": 0.0, "WeightedMatchScore": 0.0, "QualityScore": 0.0}
                strata_threshold = expert_threshold(bundle, model_pack, strata_name)
                threshold_scope = "expert" if model_pack else "strata_fallback"
                base_threshold = strata_threshold
            match_rows.append(
                {
                    "WellName": well_name,
                    "StrataName": strata_name,
                    "TargetRowCount": int(strata_mask.sum()),
                    "UsableRowCount": int(predict_mask.sum()),
                    "Stage1Threshold": strata_threshold,
                    "BaseStage1Threshold": base_threshold,
                    "ThresholdScope": threshold_scope if model_pack else "strata_fallback",
                    "EnsembleExpertCount": ensemble_info["EnsembleExpertCount"],
                    "EnsembleExpertID": ensemble_info["ExpertID"],
                    "EnsembleExpertWellSegment": ensemble_info["ExpertWellSegment"],
                    "EnsembleQualityScore": ensemble_info["QualityScore"],
                    **match,
                }
            )
            if not model_pack:
                main_df.loc[strata_mask, "FallbackReason"] = match.get("FallbackReason") or "missing_matched_expert"
                continue
            if not predict_mask.any():
                main_df.loc[strata_mask, "FallbackReason"] = "missing_required_features"
                continue
            density = gate_density_by_threshold(prob, density_raw, strata_threshold)
            main_df.loc[predict_mask, "PredFractureProb"] = prob
            main_df.loc[predict_mask, "PredDensityRaw"] = density_raw
            main_df.loc[predict_mask, "PredDensity"] = density
            main_df.loc[predict_mask, "PredHasFracture"] = (prob >= strata_threshold).astype(int)
            main_df.loc[predict_mask, "PredictionStatus"] = "predicted"
            main_df.loc[predict_mask, "ExpertID"] = ensemble_info["ExpertID"]
            main_df.loc[predict_mask, "ExpertWellSegment"] = ensemble_info["ExpertWellSegment"]
            main_df.loc[predict_mask, "MatchDistance"] = match.get("MatchDistance", np.nan)
            main_df.loc[predict_mask, "MatchScore"] = ensemble_info["MatchScore"]
            main_df.loc[predict_mask, "WeightedMatchScore"] = ensemble_info["WeightedMatchScore"]
            main_df.loc[predict_mask, "ExpertQualityScore"] = ensemble_info["QualityScore"]
            main_df.loc[predict_mask, "EnsembleExpertCount"] = ensemble_info["EnsembleExpertCount"]
            main_df.loc[predict_mask, "FallbackReason"] = match.get("FallbackReason", "")
            main_df.loc[predict_mask, "Stage1Threshold"] = strata_threshold
            main_df.loc[predict_mask, "BaseStage1Threshold"] = base_threshold
            main_df.loc[predict_mask, "ThresholdScope"] = threshold_scope
            main_df.loc[strata_mask & ~usable_mask, "FallbackReason"] = "missing_required_features"

        keep_columns = [
            *IDENTITY_COLUMNS,
            *bundle.feature_columns,
            "PredFractureProb",
            "PredDensityRaw",
            "PredDensity",
            "PredHasFracture",
            "Stage1Threshold",
            "BaseStage1Threshold",
            "ThresholdScope",
            "ExpertID",
            "ExpertWellSegment",
            "MatchDistance",
            "MatchScore",
            "WeightedMatchScore",
            "ExpertQualityScore",
            "EnsembleExpertCount",
            "PredictionStatus",
            "FallbackReason",
        ]
        pred_out = main_df[keep_columns].copy()
        pred_out["Density"] = safe_numeric(pred_out["PredDensity"]).clip(lower=0.0)
        pred_out["HasFracture"] = safe_numeric(pred_out["PredHasFracture"]).fillna(0.0).gt(0.0).astype(int)
        ordered_pred_columns = [
            *IDENTITY_COLUMNS,
            *bundle.feature_columns,
            "Density",
            "HasFracture",
            "PredFractureProb",
            "PredDensityRaw",
            "PredDensity",
            "PredHasFracture",
            "Stage1Threshold",
            "BaseStage1Threshold",
            "ThresholdScope",
            "ExpertID",
            "ExpertWellSegment",
            "MatchDistance",
            "MatchScore",
            "WeightedMatchScore",
            "ExpertQualityScore",
            "EnsembleExpertCount",
            "PredictionStatus",
            "FallbackReason",
        ]
        pred_out = pred_out[[col for col in ordered_pred_columns if col in pred_out.columns]]
        out_dir = well_output_root / well_name
        ensure_dir(out_dir)
        pred_out.to_csv(out_dir / f"{well_name}_t4_t7_density_curve.csv", index=False, encoding="utf-8-sig")
        point_parts_by_strata: list[pd.DataFrame] = []
        for strata_name in TARGET_STRATA:
            strata_pred = pred_out[pred_out["StrataName"] == strata_name].copy()
            if strata_pred.empty:
                continue
            if point_threshold is not None:
                refine_threshold = float(point_threshold)
            else:
                threshold_values = safe_numeric(strata_pred["Stage1Threshold"]).dropna()
                refine_threshold = float(threshold_values.median()) if not threshold_values.empty else float(
                    bundle.stage1_thresholds_by_strata.get(strata_name, bundle.stage1_threshold)
                )
            point_calibrator = dict(bundle.point_count_calibrators_by_strata.get(strata_name, {}))
            point_parts_by_strata.append(
                refine_points(
                    strata_pred,
                    threshold=refine_threshold,
                    min_density=point_min_density,
                    calibrator=point_calibrator,
                )
            )
        points = pd.concat(point_parts_by_strata, ignore_index=True) if point_parts_by_strata else pd.DataFrame()
        points.to_csv(out_dir / f"{well_name}_t4_t7_fracture_points.csv", index=False, encoding="utf-8-sig")
        qc = {
            "WellName": well_name,
            "Status": "ok",
            "TotalRowCount": int(len(pred_out)),
            "PredictedRowCount": int((pred_out["PredictionStatus"] == "predicted").sum()),
            "UnpredictedRowCount": int((pred_out["PredictionStatus"] != "predicted").sum()),
            "PositiveDensityRowCount": int((safe_numeric(pred_out["PredDensity"]).fillna(0.0) > 0.0).sum()),
            "PredHasFractureRowCount": int((safe_numeric(pred_out["PredHasFracture"]).fillna(0.0) > 0.0).sum()),
            "PredictedPointCount": int(len(points)),
            "MeanPredDensity": float(safe_numeric(pred_out["PredDensity"]).mean()) if pred_out["PredDensity"].notna().any() else np.nan,
            "MeanPredFractureProb": float(safe_numeric(pred_out["PredFractureProb"]).mean()) if pred_out["PredFractureProb"].notna().any() else np.nan,
        }
        pd.DataFrame([qc]).to_csv(out_dir / f"{well_name}_prediction_qc.csv", index=False, encoding="utf-8-sig")
        summary_rows.append(qc)
        prediction_parts.append(pred_out)
        point_parts.append(points)

    if prediction_parts:
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            output_dir / "all_wells_t4_t7_density_prediction.csv", index=False, encoding="utf-8-sig"
        )
    (pd.concat(point_parts, ignore_index=True) if point_parts else pd.DataFrame()).to_csv(
        output_dir / "all_wells_t4_t7_fracture_points.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary_rows).to_csv(output_dir / "all_wells_t4_t7_prediction_summary.csv", index=False, encoding="utf-8-sig")
    match_df = pd.DataFrame(match_rows)
    match_df.to_csv(output_dir / "well_strata_expert_match_manifest.csv", index=False, encoding="utf-8-sig")
    match_df.to_csv(output_dir / "selected_expert_by_well_strata.csv", index=False, encoding="utf-8-sig")
    write_delivery_summary(config, output_dir, bundle, summary_rows, match_rows)


def write_delivery_summary(
    config: dict[str, Any],
    output_dir: Path,
    bundle: ExpertLibraryBundle,
    summary_rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
) -> None:
    holdout_well = str(config.get("holdout_well") or DEFAULT_HOLDOUT_WELL)
    registry_path = output_dir / "expert_library/expert_registry.csv"
    registry_df = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    predicted_wells = [row["WellName"] for row in summary_rows if row.get("Status") == "ok"]
    prediction_csv = output_dir / "all_wells_t4_t7_density_prediction.csv"
    points_csv = output_dir / "all_wells_t4_t7_fracture_points.csv"
    prediction_df = pd.read_csv(prediction_csv, nrows=1000) if prediction_csv.exists() else pd.DataFrame()
    point_df = pd.read_csv(points_csv, nrows=1000) if points_csv.exists() else pd.DataFrame()
    required_prediction_columns = {
        "SampleID",
        "WellName",
        "X",
        "Y",
        "TIME",
        "Density",
        "HasFracture",
        "PredDensity",
        "PredFractureProb",
        "StrataName",
    }
    missing_prediction_columns = sorted(required_prediction_columns - set(prediction_df.columns))
    missing_point_columns = sorted({"SourceSampleID", "WellName", "X", "Y", "TIME", "Density", "HasFracture"} - set(point_df.columns)) if not point_df.empty else []
    holdout_in_training = bool((registry_df.get("ExpertWellSegment", pd.Series(dtype=str)) == holdout_well).any())
    predicted_row_count = int(sum(int(row.get("PredictedRowCount", 0)) for row in summary_rows))
    total_row_count = int(sum(int(row.get("TotalRowCount", 0)) for row in summary_rows))
    checks = {
        "has_prediction_csv": prediction_csv.exists(),
        "has_points_csv": points_csv.exists(),
        "prediction_has_downstream_columns": not missing_prediction_columns,
        "points_have_downstream_columns": not missing_point_columns,
        "holdout_excluded_from_training": not holdout_in_training,
        "has_predicted_rows": predicted_row_count > 0,
        "all_rows_predicted": predicted_row_count == total_row_count and total_row_count > 0,
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "status": status,
        "output_dir": str(output_dir),
        "expert_design": "strata_first_well_level_expert_library",
        "holdout_well": holdout_well,
        "holdout_in_training": holdout_in_training,
        "expert_count": int(len(registry_df)),
        "experts_by_strata": registry_df.groupby("LibraryStrata")["ExpertID"].count().to_dict() if not registry_df.empty else {},
        "predicted_well_count": int(len(predicted_wells)),
        "skipped_non_ok_well_count": int(len(pd.read_csv(output_dir / "skipped_step2_wells_manifest.csv"))) if (output_dir / "skipped_step2_wells_manifest.csv").exists() else 0,
        "total_predicted_rows": predicted_row_count,
        "total_prediction_rows": total_row_count,
        "matched_well_strata_count": int(sum(1 for row in match_rows if row.get("MatchStatus") in {"matched", "matched_without_signature_overlap"})),
        "feature_columns": bundle.feature_columns,
        "point_count_calibrators_by_strata": bundle.point_count_calibrators_by_strata,
        "downstream_contract": {
            "density_column": "Density",
            "has_fracture_column": "HasFracture",
            "prediction_required_columns": sorted(required_prediction_columns),
            "missing_prediction_columns": missing_prediction_columns,
            "missing_point_columns": missing_point_columns,
            "checks": checks,
        },
        "formal_constraints": {
            "uses_step3_formal_groups": True,
            "uses_old_step3_total_table": False,
            "density_is_main_supervision": True,
            "holdout_excluded_from_training": not holdout_in_training,
        },
    }
    write_json(output_dir / "step4_acceptance_summary.json", summary)


def main() -> int:
    args = build_parser().parse_args()
    if args.train_only and args.predict_only:
        raise SystemExit("--train-only and --predict-only cannot be used together.")
    config = read_json(Path(args.config))
    output_dir = Path(config["output_dir"])
    ensure_dir(output_dir)
    bundle: ExpertLibraryBundle | None = None
    if not args.predict_only:
        bundle = train_library(config)
    if not args.train_only:
        predict_all_wells(config, bundle=bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
