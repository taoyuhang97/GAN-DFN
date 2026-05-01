import os
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


def get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


SUPPORTED_SEGMENT_COUNT_MODES = {
    "density_integral",
    "global_density_scale",
    "global_length_rate",
    "global_prob_mass_scale",
    "global_length_prob_mean_scale",
    "global_length_prob_mass_scale",
}
SUPPORTED_SHAPE_MODES = {"density_only", "density_prob", "prob_only"}
COUNT_MODE_TO_BASIS_COL = {
    "density_integral": "RawDensityTotal",
    "global_density_scale": "RawDensityTotal",
    "global_length_rate": "SegLength",
    "global_prob_mass_scale": "ProbMass",
    "global_length_prob_mean_scale": "LengthXProbMean",
    "global_length_prob_mass_scale": "LengthXProbMass",
}
DENSITY_REQUIRED_COUNT_MODES = {"density_integral", "global_density_scale"}
DENSITY_REQUIRED_SHAPE_MODES = {"density_only", "density_prob"}


@dataclass
class PointRefineConfig:
    exist_exp_dir: str
    density_exp_dir: str
    save_dir: str
    well_names: list[str]

    depth_col: str = "TVD"
    x_col: str = "X"
    y_col: str = "Y"
    time_col: str = "TIME"

    pred_label_col: str = "PRED_LABEL"
    pred_prob_col: str = "PRED_PROB"
    gt_label_col: str = "GT_LABEL"

    density_gt_col: str = "P10"
    density_pred_col: str = "P10_PRED_RAW"

    prob_weight_gamma: float = 2.0
    use_prob_weighting: bool = True

    min_points_per_segment: int = 1
    rounding_mode: str = "round"

    density_clip_min: float = 0.0
    density_clip_max: float = 20.0

    use_pred_density: bool = True
    segment_count_mode: str = "density_integral"
    shape_mode: str = "density_prob"


def list_verify_logs(exist_exp_dir: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if not os.path.isdir(exist_exp_dir):
        raise FileNotFoundError(f"exist_exp_dir not found: {exist_exp_dir}")

    for name in os.listdir(exist_exp_dir):
        if not name.startswith("verify_"):
            continue
        well = name.replace("verify_", "", 1)
        csv_path = os.path.join(exist_exp_dir, name, "val_pred_full_log.csv")
        if os.path.exists(csv_path):
            result[well] = csv_path
    if not result:
        raise FileNotFoundError(f"No verify_* logs found under: {exist_exp_dir}")
    return result


def load_density_pred_series(density_exp_dir: str, well: str, density_pred_col: str) -> pd.DataFrame:
    csv_path = os.path.join(density_exp_dir, well, "val_density_pred_full_log.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Density log not found for {well}: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "RAW_ROW_IDX" not in df.columns:
        raise ValueError(f"Density log missing RAW_ROW_IDX: {csv_path}")
    if density_pred_col not in df.columns:
        fallback = None
        if density_pred_col.endswith("_PRED_RAW"):
            cand = density_pred_col.replace("_PRED_RAW", "_PRED")
            if cand in df.columns:
                fallback = cand
        elif density_pred_col.endswith("_PRED"):
            cand = density_pred_col.replace("_PRED", "_PRED_RAW")
            if cand in df.columns:
                fallback = cand
        if fallback is None:
            raise ValueError(f"Density log missing {density_pred_col}: {csv_path}")
        density_pred_col = fallback

    out = df[["RAW_ROW_IDX", density_pred_col]].copy()
    out["RAW_ROW_IDX"] = safe_to_numeric(out["RAW_ROW_IDX"]).astype("Int64")
    out[density_pred_col] = safe_to_numeric(out[density_pred_col])
    return out


def find_binary_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def integrate_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    cum = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return cum
    dx = np.diff(x)
    dx = np.clip(dx, a_min=0.0, a_max=None)
    area = 0.5 * (y[1:] + y[:-1]) * dx
    cum[1:] = np.cumsum(area)
    return cum


def integrate_total(x: np.ndarray, y: np.ndarray | None) -> float:
    if y is None or len(x) <= 1:
        return 0.0
    return float(integrate_trapezoid(x, y)[-1])


def pick_points_by_equal_intensity(
    depth: np.ndarray,
    intensity: np.ndarray,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    assert len(depth) == len(intensity)
    if n_points <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
    if len(depth) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
    if len(depth) == 1:
        return np.array([float(depth[0])]), np.array([0], dtype=np.int64)
    n_points = int(min(n_points, len(depth)))

    cum = integrate_trapezoid(depth, intensity)
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 0:
        mid = len(depth) // 2
        return np.array([float(depth[mid])]), np.array([mid], dtype=np.int64)

    step = total / n_points
    targets = (np.arange(n_points, dtype=np.float64) + 0.5) * step
    picked_depths = np.zeros(n_points, dtype=np.float64)
    picked_idx = np.zeros(n_points, dtype=np.int64)

    for i, t in enumerate(targets):
        idx = int(np.searchsorted(cum, t, side="left"))
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
            ratio = (t - c0) / (c1 - c0)
            picked_depths[i] = d0 + ratio * (d1 - d0)

        picked_idx[i] = idx if abs(depth[idx] - picked_depths[i]) < abs(depth[idx - 1] - picked_depths[i]) else idx - 1

    return picked_depths, picked_idx


def compute_n_points(total_intensity: float, min_points: int, rounding_mode: str) -> int:
    if not np.isfinite(total_intensity) or total_intensity <= 0:
        return 0
    if rounding_mode == "floor":
        n = int(np.floor(total_intensity))
    elif rounding_mode == "ceil":
        n = int(np.ceil(total_intensity))
    elif rounding_mode == "round":
        n = int(np.round(total_intensity))
    else:
        raise ValueError(f"Unsupported rounding_mode: {rounding_mode}")
    return max(min_points, n)


def validate_segment_count_mode(mode: str) -> str:
    if mode not in SUPPORTED_SEGMENT_COUNT_MODES:
        raise ValueError(f"Unsupported segment_count_mode: {mode}")
    return mode


def validate_shape_mode(mode: str) -> str:
    if mode not in SUPPORTED_SHAPE_MODES:
        raise ValueError(f"Unsupported shape_mode: {mode}")
    return mode


def pred_density_required(config: PointRefineConfig) -> bool:
    return (
        config.segment_count_mode in DENSITY_REQUIRED_COUNT_MODES
        or config.shape_mode in DENSITY_REQUIRED_SHAPE_MODES
    )


def compute_shape_intensity(
    density_seg: np.ndarray | None,
    prob_seg: np.ndarray | None,
    config: PointRefineConfig,
    shape_mode: str,
) -> np.ndarray:
    if prob_seg is not None:
        prob_seg = np.clip(prob_seg, 0.0, 1.0)
        prob_gamma = np.power(prob_seg, config.prob_weight_gamma)
    else:
        prob_gamma = None

    if shape_mode == "density_only":
        if density_seg is None:
            raise ValueError("shape_mode=density_only requires density input")
        intensity = density_seg.copy()
    elif shape_mode == "density_prob":
        if density_seg is None:
            raise ValueError("shape_mode=density_prob requires density input")
        intensity = density_seg.copy()
        if prob_gamma is not None:
            w_mean = float(np.mean(prob_gamma)) if prob_gamma.size > 0 else 0.0
            if w_mean > 1e-8:
                intensity = intensity * (prob_gamma / w_mean)
    elif shape_mode == "prob_only":
        base_len = len(prob_seg) if prob_seg is not None else len(density_seg)
        if base_len <= 0:
            return np.array([], dtype=np.float64)
        if prob_gamma is not None:
            w_mean = float(np.mean(prob_gamma)) if prob_gamma.size > 0 else 0.0
            if w_mean > 1e-8:
                intensity = prob_gamma / w_mean
            else:
                intensity = np.ones(base_len, dtype=np.float64)
        else:
            intensity = np.ones(base_len, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported shape_mode: {shape_mode}")

    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(intensity.astype(np.float64), a_min=0.0, a_max=None)


def extract_segment_payloads(
    df: pd.DataFrame,
    mask_col: str,
    density_col: str | None,
    config: PointRefineConfig,
    prob_col: str | None = None,
    shape_mode: str | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    df = df.copy()
    actual_shape_mode = shape_mode or config.shape_mode

    if config.depth_col not in df.columns:
        raise ValueError(f"Missing depth col {config.depth_col}")
    if mask_col not in df.columns:
        raise ValueError(f"Missing mask col {mask_col}")
    if density_col is not None and density_col not in df.columns:
        raise ValueError(f"Missing density col {density_col}")
    if prob_col is not None and prob_col not in df.columns:
        raise ValueError(f"Missing prob col {prob_col}")

    df[config.depth_col] = safe_to_numeric(df[config.depth_col])
    if density_col is not None:
        df[density_col] = safe_to_numeric(df[density_col])
    if prob_col is not None:
        df[prob_col] = safe_to_numeric(df[prob_col])

    df = df.sort_values(config.depth_col).reset_index(drop=True)

    depth = df[config.depth_col].to_numpy(dtype=np.float64)
    mask = safe_to_numeric(df[mask_col]).fillna(0).astype(int).to_numpy(dtype=np.int8) == 1

    density = None
    if density_col is not None:
        density = df[density_col].fillna(0).to_numpy(dtype=np.float64)
        density = np.clip(density, config.density_clip_min, config.density_clip_max)
        density[~np.isfinite(density)] = 0.0
        density = np.clip(density, a_min=0.0, a_max=None)
        density = density * mask.astype(np.float64)

    prob = None
    if prob_col is not None:
        prob = df[prob_col].fillna(0).to_numpy(dtype=np.float64)
        prob = np.clip(prob, 0.0, 1.0)

    payloads: list[dict] = []
    for seg_id, (s, e) in enumerate(find_binary_segments(mask), start=1):
        depth_seg = depth[s : e + 1]
        density_seg = density[s : e + 1] if density is not None else None
        prob_seg = prob[s : e + 1] if prob is not None else None
        prob_gamma_seg = np.power(prob_seg, config.prob_weight_gamma) if prob_seg is not None else None
        shape_intensity = compute_shape_intensity(density_seg, prob_seg, config, actual_shape_mode)

        payloads.append(
            {
                "Segment_ID": seg_id,
                "StartIdx": int(s),
                "EndIdx": int(e),
                "SegStartDepth": float(depth_seg[0]) if depth_seg.size else np.nan,
                "SegEndDepth": float(depth_seg[-1]) if depth_seg.size else np.nan,
                "SegLength": float(max(depth_seg[-1] - depth_seg[0], 0.0)) if depth_seg.size >= 2 else 0.0,
                "RawDensityTotal": integrate_total(depth_seg, density_seg),
                "ProbMean": float(np.mean(prob_seg)) if prob_seg is not None and prob_seg.size > 0 else np.nan,
                "ProbMax": float(np.max(prob_seg)) if prob_seg is not None and prob_seg.size > 0 else np.nan,
                "ProbGammaMean": float(np.mean(prob_gamma_seg)) if prob_gamma_seg is not None and prob_gamma_seg.size > 0 else np.nan,
                "ProbMass": integrate_total(depth_seg, prob_gamma_seg),
                "DensityMean": float(np.mean(density_seg)) if density_seg is not None and density_seg.size > 0 else np.nan,
                "depth_seg": depth_seg,
                "density_seg": density_seg,
                "shape_intensity": shape_intensity,
            }
        )

    return df, payloads


def build_fracture_points(
    df: pd.DataFrame,
    mask_col: str,
    density_col: str,
    config: PointRefineConfig,
    prob_col: str | None = None,
    use_prob_weighting: bool = False,
) -> pd.DataFrame:
    shape_mode = "density_prob" if use_prob_weighting else "density_only"
    df_sorted, payloads = extract_segment_payloads(
        df,
        mask_col=mask_col,
        density_col=density_col,
        config=config,
        prob_col=prob_col,
        shape_mode=shape_mode,
    )

    all_points: list[dict] = []
    for payload in payloads:
        n_points = compute_n_points(payload["RawDensityTotal"], config.min_points_per_segment, config.rounding_mode)
        if n_points <= 0:
            continue

        picked_depths, picked_rel_idx = pick_points_by_equal_intensity(
            payload["depth_seg"],
            payload["shape_intensity"],
            n_points,
        )
        for k, (d_pick, rel_idx) in enumerate(zip(picked_depths, picked_rel_idx), start=1):
            abs_idx = int(payload["StartIdx"] + rel_idx)
            row = df_sorted.iloc[abs_idx]
            all_points.append(
                {
                    "Segment_ID": payload["Segment_ID"],
                    "Point_ID_In_Segment": k,
                    config.depth_col: float(d_pick),
                    "NearestSampleDepth": float(row[config.depth_col]),
                    config.x_col: float(row[config.x_col]) if config.x_col in df_sorted.columns else np.nan,
                    config.y_col: float(row[config.y_col]) if config.y_col in df_sorted.columns else np.nan,
                    config.time_col: float(row[config.time_col]) if config.time_col in df_sorted.columns else np.nan,
                    "SegStartDepth": payload["SegStartDepth"],
                    "SegEndDepth": payload["SegEndDepth"],
                    "SegLength": payload["SegLength"],
                    "SegTotalIntensity": payload["RawDensityTotal"],
                    "SegNPoints": int(n_points),
                    "SegMeanDensity": payload["DensityMean"],
                }
            )

    return pd.DataFrame(all_points)


def count_points_in_depth_range(point_depths: np.ndarray, start_depth: float, end_depth: float) -> int:
    if point_depths.size == 0:
        return 0
    return int(((point_depths >= start_depth) & (point_depths <= end_depth)).sum())


def payload_to_segment_row(payload: dict) -> dict:
    return {
        "Segment_ID": payload["Segment_ID"],
        "SegStartDepth": payload["SegStartDepth"],
        "SegEndDepth": payload["SegEndDepth"],
        "SegLength": payload["SegLength"],
        "RawDensityTotal": payload["RawDensityTotal"],
        "DensityMean": payload["DensityMean"],
        "ProbMean": payload["ProbMean"],
        "ProbMax": payload["ProbMax"],
        "ProbGammaMean": payload["ProbGammaMean"],
        "ProbMass": payload["ProbMass"],
        "LengthXProbMean": payload["SegLength"] * max(payload["ProbMean"], 0.0)
        if np.isfinite(payload["ProbMean"])
        else 0.0,
        "LengthXProbMass": payload["SegLength"] * max(payload["ProbMass"], 0.0)
        if np.isfinite(payload["ProbMass"])
        else 0.0,
    }


def get_payload_basis_value(payload: dict, basis_col: str) -> float:
    if basis_col in payload:
        basis_value = payload[basis_col]
    else:
        basis_value = payload_to_segment_row(payload).get(basis_col, 0.0)
    if not np.isfinite(basis_value):
        return 0.0
    return float(max(basis_value, 0.0))


def build_calibration_segment_frame(well_data: dict[str, dict], calib_wells: list[str], config: PointRefineConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for well in calib_wells:
        info = well_data[well]
        gt_points = build_fracture_points(
            info["df"],
            mask_col=config.gt_label_col,
            density_col=config.density_gt_col,
            config=config,
            prob_col=None,
            use_prob_weighting=False,
        )
        gt_depths = gt_points[config.depth_col].to_numpy(dtype=np.float64) if not gt_points.empty else np.array([])
        _, payloads = extract_segment_payloads(
            info["df"],
            mask_col=config.pred_label_col,
            density_col=info["pred_density_col"],
            config=config,
            prob_col=config.pred_prob_col,
            shape_mode=config.shape_mode,
        )
        for payload in payloads:
            row = payload_to_segment_row(payload)
            row["well"] = well
            row["GTPointCountInPredSegment"] = count_points_in_depth_range(
                gt_depths,
                payload["SegStartDepth"],
                payload["SegEndDepth"],
            )
            rows.append(row)
    return pd.DataFrame(rows)


def fit_segment_count_calibrator(calib_seg_df: pd.DataFrame, segment_count_mode: str, calib_wells: list[str]) -> dict:
    basis_col = COUNT_MODE_TO_BASIS_COL[segment_count_mode]
    if segment_count_mode == "density_integral":
        return {
            "mode": segment_count_mode,
            "basis_col": basis_col,
            "scale": 1.0,
            "has_calibration": False,
            "num_segments": int(len(calib_seg_df)),
            "num_wells": int(len(calib_wells)),
            "basis_sum": float("nan"),
            "target_sum": float("nan"),
        }

    if calib_seg_df.empty:
        return {
            "mode": segment_count_mode,
            "basis_col": basis_col,
            "scale": 1.0,
            "has_calibration": False,
            "num_segments": 0,
            "num_wells": int(len(calib_wells)),
            "basis_sum": float("nan"),
            "target_sum": float("nan"),
        }

    basis = safe_to_numeric(calib_seg_df[basis_col]).fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    target = (
        safe_to_numeric(calib_seg_df["GTPointCountInPredSegment"])
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )
    basis_sum = float(np.sum(basis))
    target_sum = float(np.sum(target))
    if basis_sum > 1e-8:
        scale = target_sum / basis_sum
        has_calibration = True
    else:
        scale = 1.0
        has_calibration = False

    return {
        "mode": segment_count_mode,
        "basis_col": basis_col,
        "scale": float(scale),
        "has_calibration": has_calibration,
        "num_segments": int(len(calib_seg_df)),
        "num_wells": int(len(calib_wells)),
        "basis_sum": basis_sum,
        "target_sum": target_sum,
    }


def resolve_segment_count_value(payload: dict, calibrator: dict) -> tuple[float, float]:
    basis_col = calibrator["basis_col"]
    basis_value = get_payload_basis_value(payload, basis_col)
    count_value = basis_value * float(calibrator["scale"])
    return basis_value, float(max(count_value, 0.0))


def build_pred_fracture_points(
    df: pd.DataFrame,
    pred_density_col: str | None,
    config: PointRefineConfig,
    calibrator: dict,
    gt_depths: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted, payloads = extract_segment_payloads(
        df,
        mask_col=config.pred_label_col,
        density_col=pred_density_col,
        config=config,
        prob_col=config.pred_prob_col,
        shape_mode=config.shape_mode,
    )

    all_points: list[dict] = []
    segment_rows: list[dict] = []

    for payload in payloads:
        basis_value, count_value = resolve_segment_count_value(payload, calibrator)
        n_points = compute_n_points(count_value, config.min_points_per_segment, config.rounding_mode)
        gt_count = count_points_in_depth_range(
            gt_depths,
            payload["SegStartDepth"],
            payload["SegEndDepth"],
        )

        segment_row = payload_to_segment_row(payload)
        segment_row.update(
            {
                "CountBasisCol": calibrator["basis_col"],
                "CountBasisValue": basis_value,
                "CalibratedCountValue": count_value,
                "PredPointCount": int(n_points),
                "GTPointCountInPredSegment": int(gt_count),
                "SegmentCountMode": config.segment_count_mode,
                "ShapeMode": config.shape_mode,
            }
        )
        segment_rows.append(segment_row)

        if n_points <= 0:
            continue

        picked_depths, picked_rel_idx = pick_points_by_equal_intensity(
            payload["depth_seg"],
            payload["shape_intensity"],
            n_points,
        )
        for k, (d_pick, rel_idx) in enumerate(zip(picked_depths, picked_rel_idx), start=1):
            abs_idx = int(payload["StartIdx"] + rel_idx)
            row = df_sorted.iloc[abs_idx]
            all_points.append(
                {
                    "Segment_ID": payload["Segment_ID"],
                    "Point_ID_In_Segment": k,
                    config.depth_col: float(d_pick),
                    "NearestSampleDepth": float(row[config.depth_col]),
                    config.x_col: float(row[config.x_col]) if config.x_col in df_sorted.columns else np.nan,
                    config.y_col: float(row[config.y_col]) if config.y_col in df_sorted.columns else np.nan,
                    config.time_col: float(row[config.time_col]) if config.time_col in df_sorted.columns else np.nan,
                    "SegStartDepth": payload["SegStartDepth"],
                    "SegEndDepth": payload["SegEndDepth"],
                    "SegLength": payload["SegLength"],
                    "SegTotalIntensity": payload["RawDensityTotal"],
                    "SegNPoints": int(n_points),
                    "SegMeanDensity": payload["DensityMean"],
                    "CountBasisCol": calibrator["basis_col"],
                    "CountBasisValue": basis_value,
                    "CalibratedCountValue": count_value,
                    "SegmentCountMode": config.segment_count_mode,
                    "ShapeMode": config.shape_mode,
                    "GTPointCountInPredSegment": int(gt_count),
                }
            )

    return pd.DataFrame(all_points), pd.DataFrame(segment_rows)


def nearest_distance_stats(src_depths: np.ndarray, ref_depths: np.ndarray) -> dict[str, float]:
    if src_depths.size == 0 or ref_depths.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    src = np.sort(src_depths.astype(np.float64))
    ref = np.sort(ref_depths.astype(np.float64))

    dists = np.zeros_like(src)
    j = 0
    for i, d in enumerate(src):
        while j + 1 < ref.size and abs(ref[j + 1] - d) <= abs(ref[j] - d):
            j += 1
        dists[i] = abs(ref[j] - d)

    dists_sorted = np.sort(dists)
    p90 = float(dists_sorted[int(np.floor(0.9 * (len(dists_sorted) - 1)))])
    return {
        "mean": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "p90": p90,
        "max": float(np.max(dists)),
    }


def main() -> None:
    default_exist_exp_dir = (
        r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝存在性预测/LSTM/单井验证/成像测井裂缝预测/cnn+lstm"
        r"\exp40_missing_drop_ac_gr_3x3_seq5_AC_GR"
    )
    default_density_exp_dir = (
        r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝位置预测/基于密度的裂缝点位分析/裂缝密度预测"
        r"\cnn+lstm_p10_only_scaled"
    )
    default_save_dir = (
        r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝位置预测/基于密度的裂缝点位分析/裂缝点位精细化"
        r"\density_integral"
    )

    exist_exp_dir = get_env_str("EXP_EXIST_EXP_DIR", default_exist_exp_dir)
    density_exp_dir = get_env_str("EXP_DENSITY_EXP_DIR", default_density_exp_dir)
    save_dir = get_env_str("EXP_SAVE_DIR", default_save_dir)

    use_pred_density = get_env_bool("EXP_USE_PRED_DENSITY", True)
    use_prob_weighting = get_env_bool("EXP_USE_PROB_WEIGHTING", True)
    prob_weight_gamma = get_env_float("EXP_PROB_WEIGHT_GAMMA", 2.0)
    min_points_per_segment = get_env_int("EXP_MIN_POINTS_PER_SEGMENT", 1)
    rounding_mode = get_env_str("EXP_ROUNDING_MODE", "round")
    density_clip_max = get_env_float("EXP_DENSITY_CLIP_MAX", 20.0)
    density_gt_col = get_env_str("EXP_DENSITY_GT_COL", "P10")
    density_pred_col = get_env_str("EXP_DENSITY_PRED_COL", "P10_PRED_RAW")
    segment_count_mode = validate_segment_count_mode(get_env_str("EXP_SEGMENT_COUNT_MODE", "density_integral"))
    default_shape_mode = "density_prob" if use_prob_weighting else "density_only"
    shape_mode = validate_shape_mode(get_env_str("EXP_SHAPE_MODE", default_shape_mode))

    verify_logs = list_verify_logs(exist_exp_dir)
    well_names_env = os.getenv("EXP_WELL_NAMES")
    if well_names_env:
        well_names = [w.strip() for w in well_names_env.split(",") if w.strip()]
    else:
        well_names = sorted(verify_logs.keys())

    config = PointRefineConfig(
        exist_exp_dir=exist_exp_dir,
        density_exp_dir=density_exp_dir,
        save_dir=save_dir,
        well_names=well_names,
        use_pred_density=use_pred_density,
        use_prob_weighting=use_prob_weighting,
        prob_weight_gamma=prob_weight_gamma,
        min_points_per_segment=min_points_per_segment,
        rounding_mode=rounding_mode,
        density_clip_max=density_clip_max,
        density_gt_col=density_gt_col,
        density_pred_col=density_pred_col,
        segment_count_mode=segment_count_mode,
        shape_mode=shape_mode,
    )

    os.makedirs(config.save_dir, exist_ok=True)

    well_data: dict[str, dict] = {}
    need_pred_density = pred_density_required(config)
    for well in config.well_names:
        if well not in verify_logs:
            print(f"[WARN] well not found in exist logs, skip: {well}")
            continue

        exist_csv = verify_logs[well]
        df = pd.read_csv(exist_csv, encoding="utf-8-sig")
        if "RAW_ROW_IDX" not in df.columns:
            raise ValueError(f"Exist log missing RAW_ROW_IDX: {exist_csv}")
        df["RAW_ROW_IDX"] = safe_to_numeric(df["RAW_ROW_IDX"]).astype("Int64")

        pred_density_col_for_well = None
        if need_pred_density:
            if config.use_pred_density:
                den_df = load_density_pred_series(config.density_exp_dir, well, config.density_pred_col)
                df = df.merge(den_df, on="RAW_ROW_IDX", how="left", suffixes=("", "_DEN"))
                if config.density_pred_col not in df.columns:
                    raise ValueError(f"After merge, missing {config.density_pred_col} for well {well}")
                pred_density_col_for_well = config.density_pred_col
            else:
                pred_density_col_for_well = config.density_gt_col

        well_data[well] = {
            "df": df,
            "exist_csv": exist_csv,
            "pred_density_col": pred_density_col_for_well,
        }

    active_wells = [well for well in config.well_names if well in well_data]
    summary_rows: list[dict] = []

    for well in active_wells:
        info = well_data[well]
        df = info["df"]

        gt_points = build_fracture_points(
            df,
            mask_col=config.gt_label_col,
            density_col=config.density_gt_col,
            config=config,
            prob_col=None,
            use_prob_weighting=False,
        )
        gt_depths = gt_points[config.depth_col].to_numpy(dtype=np.float64) if not gt_points.empty else np.array([])

        calib_wells = [other_well for other_well in active_wells if other_well != well]
        calib_seg_df = build_calibration_segment_frame(well_data, calib_wells, config)
        calibrator = fit_segment_count_calibrator(calib_seg_df, config.segment_count_mode, calib_wells)

        pred_points, pred_seg_df = build_pred_fracture_points(
            df,
            pred_density_col=info["pred_density_col"],
            config=config,
            calibrator=calibrator,
            gt_depths=gt_depths,
        )

        gt_depths = gt_points[config.depth_col].to_numpy(dtype=np.float64) if not gt_points.empty else np.array([])
        pred_depths = pred_points[config.depth_col].to_numpy(dtype=np.float64) if not pred_points.empty else np.array([])

        pred_to_gt = nearest_distance_stats(pred_depths, gt_depths)
        gt_to_pred = nearest_distance_stats(gt_depths, pred_depths)

        out_dir = os.path.join(config.save_dir, well)
        os.makedirs(out_dir, exist_ok=True)

        gt_points.to_csv(os.path.join(out_dir, "gt_fracture_points.csv"), index=False, encoding="utf-8-sig")
        pred_points.to_csv(os.path.join(out_dir, "pred_fracture_points.csv"), index=False, encoding="utf-8-sig")
        pred_seg_df.to_csv(os.path.join(out_dir, "pred_segment_summary.csv"), index=False, encoding="utf-8-sig")
        calib_seg_df.to_csv(os.path.join(out_dir, "count_calibration_segments.csv"), index=False, encoding="utf-8-sig")

        metrics = {
            "well": well,
            "exist_csv": info["exist_csv"],
            "density_exp_dir": config.density_exp_dir,
            "use_pred_density": config.use_pred_density,
            "use_prob_weighting": config.use_prob_weighting,
            "prob_weight_gamma": config.prob_weight_gamma,
            "min_points_per_segment": config.min_points_per_segment,
            "rounding_mode": config.rounding_mode,
            "density_clip_max": config.density_clip_max,
            "segment_count_mode": config.segment_count_mode,
            "shape_mode": config.shape_mode,
            "count_calibration_basis_col": calibrator["basis_col"],
            "count_calibration_scale": calibrator["scale"],
            "count_calibration_has_fit": calibrator["has_calibration"],
            "count_calibration_num_segments": calibrator["num_segments"],
            "count_calibration_num_wells": calibrator["num_wells"],
            "count_calibration_basis_sum": calibrator["basis_sum"],
            "count_calibration_target_sum": calibrator["target_sum"],
            "N_gt_points": int(len(gt_depths)),
            "N_pred_points": int(len(pred_depths)),
            "count_diff": int(len(pred_depths) - len(gt_depths)),
            "pred_to_gt_mean_dist": pred_to_gt["mean"],
            "pred_to_gt_median_dist": pred_to_gt["median"],
            "pred_to_gt_p90_dist": pred_to_gt["p90"],
            "gt_to_pred_mean_dist": gt_to_pred["mean"],
            "gt_to_pred_median_dist": gt_to_pred["median"],
            "gt_to_pred_p90_dist": gt_to_pred["p90"],
        }

        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as file_obj:
            json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

        summary_rows.append(metrics)

        print(
            f"\n[{well}] count_mode={config.segment_count_mode} shape_mode={config.shape_mode} "
            f"scale={calibrator['scale']:.4f} gt_points={metrics['N_gt_points']} "
            f"pred_points={metrics['N_pred_points']} diff={metrics['count_diff']} "
            f"| pred->gt mean={metrics['pred_to_gt_mean_dist']:.3f}m "
            f"p90={metrics['pred_to_gt_p90_dist']:.3f}m"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(config.save_dir, "fracture_point_refine_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with open(os.path.join(config.save_dir, "config.json"), "w", encoding="utf-8") as file_obj:
        json.dump(config.__dict__, file_obj, ensure_ascii=False, indent=2)

    print(f"\nSummary saved: {summary_csv}")


if __name__ == "__main__":
    main()
