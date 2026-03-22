from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import joblib
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import build_segment_count_dataset as dataset_builder

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


ROOT = Path(__file__).resolve().parent
DEFAULT_EXIST_EXP_DIR = Path(
    r"E:/项目/石油项目/断缝储/原始数据/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝存在性预测/LSTM/单井验证/成像测井裂缝预测/cnn+lstm/exp40_missing_drop_ac_gr_3x3_seq5_AC_GR"
)
DEFAULT_SAMPLE_DIR = Path(
    r"E:/项目/石油项目/断缝储/原始数据/wx数据/砂砾岩/优化阶段一/研究内容一/成像测井/裂缝样本"
)
DEFAULT_RAW_LABEL_DIR = Path(
    r"E:/项目/石油项目/断缝储/原始数据/wx数据/砂砾岩/研究内容一/成像测井/裂缝标注"
)
BASE_SAVE_DIR = Path(
    r"E:/项目/石油项目/断缝储/原始数据/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝位置预测/基于密度的裂缝点位分析/裂缝点位精细化"
)
DEFAULT_DOCX_PATH = Path(r"D:/项目/石油开采/断缝储实验/实验记录20260319.docx")
DOC_TITLE = "实验记录 20260319"

WELL_TO_RAW_FILE = {
    "车660-1": "车660_1.xlsx",
    "车660-2": "车660_2.xlsx",
    "车662": "车662.xlsx",
    "车663": "车663.xlsx",
}
WELL_TO_SAMPLE_FILE = {
    "车660-1": "车660-1_sample.csv",
    "车660-2": "车660-2_sample.csv",
    "车662": "车662_sample.csv",
    "车663": "车663_sample.csv",
}
ANALYSIS_ONLY_FEATURES = {
    "GTPointCountInPredSegment",
    "RawGTCountInPredSegment",
    "HasRawPoint",
    "GTDevOverlapLen",
    "GTDevOverlapRatio",
    "GTDevOverlapCount",
    "GTDevOverlapFlag",
    "TargetWeight",
    "TargetLabelType",
}
DEFAULT_RESULT_HEADERS = [
    "Well",
    "GT points",
    "Pred points",
    "Count diff",
    "Count err %",
    "pred->gt mean",
    "pred->gt p90",
    "gt->pred mean",
    "gt->pred p90",
    "SegMAE",
    "CoveredRaw",
    "MissedRaw",
]
DEFAULT_RESULT_KEYS = [
    "well",
    "N_gt_points",
    "N_pred_points",
    "count_diff",
    "count_error_pct",
    "pred_to_gt_mean_dist",
    "pred_to_gt_p90_dist",
    "gt_to_pred_mean_dist",
    "gt_to_pred_p90_dist",
    "segment_count_mae",
    "raw_points_covered_by_pred_segments",
    "raw_points_missed_outside_pred_segments",
]
MODEL_ARTIFACT_FILENAME = "segment_refine_model.joblib"
MODEL_META_FILENAME = "segment_refine_model_meta.json"
BOUNDARY_EXPAND_EFFECTIVE_MASK_COL = "__PRED_LABEL_BOUNDARY_EXPANDED"


def sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text).strip("_")


def parse_list_arg(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def format_verify_well_dir_name(well_name: str) -> str:
    well_name = str(well_name).strip()
    return well_name if well_name.startswith("verify_") else f"verify_{well_name}"


def validate_train_distance_weight_mode(mode: str) -> str:
    normalized = str(mode or "none").strip().lower()
    valid_modes = {"none", "inverse"}
    if normalized not in valid_modes:
        raise ValueError(f"Unsupported train_distance_weight_mode: {mode}, available={sorted(valid_modes)}")
    return normalized


def infer_train_distance_csv_from_exist_exp_dir(exist_exp_dir: Path) -> Path | None:
    exist_exp_dir = Path(exist_exp_dir)
    if not exist_exp_dir.exists():
        return None
    for verify_dir in sorted(exist_exp_dir.glob("verify_*")):
        config_path = verify_dir / "config.json"
        if not config_path.exists():
            continue
        try:
            with config_path.open("r", encoding="utf-8-sig") as file_obj:
                config = json.load(file_obj)
        except Exception:
            continue
        distance_path = str(config.get("DIST_MATRIX_PATH") or "").strip()
        if distance_path:
            return Path(distance_path)
    return None


def load_train_distance_matrix(distance_csv: Path) -> pd.DataFrame:
    distance_csv = Path(distance_csv)
    if not distance_csv.exists():
        raise FileNotFoundError(f"Train distance CSV not found: {distance_csv}")
    distance_df = pd.read_csv(distance_csv, encoding="utf-8-sig", index_col=0)
    if distance_df.empty:
        raise ValueError(f"Train distance CSV is empty: {distance_csv}")
    distance_df.index = [str(idx).strip() for idx in distance_df.index]
    distance_df.columns = [str(col).strip() for col in distance_df.columns]
    distance_df = distance_df.apply(pd.to_numeric, errors="coerce")
    return distance_df


def resolve_train_distance_resources(args) -> tuple[Path | None, pd.DataFrame | None]:
    mode = validate_train_distance_weight_mode(str(args.train_distance_weight_mode))
    if mode == "none":
        return None, None
    train_distance_csv = str(args.train_distance_csv or "").strip()
    resolved_path = Path(train_distance_csv) if train_distance_csv else infer_train_distance_csv_from_exist_exp_dir(Path(args.exist_exp_dir))
    if resolved_path is None:
        raise ValueError(
            "train_distance_weight_mode is enabled but no distance CSV was provided or inferred from exist_exp_dir"
        )
    return resolved_path, load_train_distance_matrix(resolved_path)


def build_train_distance_weight_info(
    target_well: str,
    train_wells: list[str],
    train_distance_weight_mode: str,
    distance_df: pd.DataFrame | None,
    weight_power: float,
    weight_min: float,
    weight_max: float,
) -> dict:
    well_distances = {str(well): np.nan for well in train_wells}
    well_weights = {str(well): 1.0 for well in train_wells}
    info = {
        "mode": train_distance_weight_mode,
        "target_well": str(target_well),
        "well_distances": well_distances,
        "well_weights": well_weights,
        "weight_power": float(weight_power),
        "weight_min": float(weight_min),
        "weight_max": float(weight_max),
        "raw_similarity_mean": np.nan,
    }
    if train_distance_weight_mode == "none" or not train_wells:
        return info
    if distance_df is None:
        raise ValueError("distance_df is required when train_distance_weight_mode is enabled")
    if target_well not in distance_df.index:
        raise ValueError(f"Target well {target_well} not found in train distance matrix")
    missing_train_wells = [well for well in train_wells if well not in distance_df.columns]
    if missing_train_wells:
        raise ValueError(f"Train wells missing in train distance matrix: {missing_train_wells}")

    distances = distance_df.loc[target_well, train_wells]
    distances = pd.to_numeric(distances, errors="coerce")
    if distances.isna().any():
        missing_values = distances[distances.isna()].index.tolist()
        raise ValueError(f"Invalid train distance values for target well {target_well}: {missing_values}")

    clipped_power = max(float(weight_power), 1e-6)
    raw_similarity = 1.0 / np.power(np.maximum(distances.to_numpy(dtype=np.float64), 1e-6), clipped_power)
    similarity_mean = float(np.mean(raw_similarity)) if raw_similarity.size else 1.0
    normalized_weights = raw_similarity / max(similarity_mean, 1e-6)
    clipped_weights = np.clip(normalized_weights, float(weight_min), float(weight_max))

    info["well_distances"] = {str(well): float(distances.loc[well]) for well in train_wells}
    info["well_weights"] = {str(well): float(clipped_weights[idx]) for idx, well in enumerate(train_wells)}
    info["raw_similarity_mean"] = similarity_mean
    return info


def apply_train_distance_weight_info(train_df: pd.DataFrame, distance_weight_info: dict) -> pd.DataFrame:
    out = train_df.copy()
    well_distances = dict(distance_weight_info.get("well_distances", {}))
    well_weights = dict(distance_weight_info.get("well_weights", {}))
    out["TrainDistanceTargetWell"] = str(distance_weight_info.get("target_well", ""))
    out["TrainDistanceToTargetWell"] = out["WellName"].map(well_distances)
    out["TrainWellDistanceWeight"] = out["WellName"].map(well_weights).fillna(1.0).astype(np.float64)
    out["TrainSampleWeightUsed"] = (
        out["TargetWeight"].fillna(1.0).astype(np.float64) * out["TrainWellDistanceWeight"].astype(np.float64)
    )
    return out


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def validate_optional_count_mode(mode: str) -> str:
    normalized = str(mode or "").strip()
    if not normalized:
        return ""
    if normalized not in dataset_builder.POINT.COUNT_MODE_TO_BASIS_COL:
        raise ValueError(
            f"Unsupported fallback_count_mode: {normalized}, "
            f"available={sorted(dataset_builder.POINT.COUNT_MODE_TO_BASIS_COL.keys())}"
        )
    return normalized


def validate_global_post_scale_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if not normalized:
        return "none"
    if normalized not in {"none", "train_sum_match", "train_sum_match_up_only"}:
        raise ValueError(
            f"Unsupported global_post_scale_mode: {normalized}, "
            "available=['none', 'train_sum_match', 'train_sum_match_up_only']"
        )
    return normalized


def validate_selective_post_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if not normalized:
        return "none"
    if normalized not in {"none", "train_zero_quantile_downscale", "train_zero_dual_quantile_downscale"}:
        raise ValueError(
            f"Unsupported selective_post_mode: {normalized}, "
            "available=['none', 'train_zero_quantile_downscale', 'train_zero_dual_quantile_downscale']"
        )
    return normalized


def validate_boundary_expand_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if not normalized:
        return "none"
    if normalized not in {"none", "prob_shoulder"}:
        raise ValueError(
            f"Unsupported boundary_expand_mode: {normalized}, "
            "available=['none', 'prob_shoulder']"
        )
    return normalized


def resolve_feature_cols(feature_preset: str, feature_cols_raw: str) -> list[str]:
    if feature_preset:
        if feature_preset not in dataset_builder.FEATURE_PRESETS:
            raise ValueError(
                f"Unsupported feature_preset: {feature_preset}, "
                f"available={sorted(dataset_builder.FEATURE_PRESETS.keys())}"
            )
        return list(dataset_builder.FEATURE_PRESETS[feature_preset])
    return parse_list_arg(feature_cols_raw)


def validate_feature_cols(feature_cols: list[str], dataset_df: pd.DataFrame) -> None:
    missing_features = [feature for feature in feature_cols if feature not in dataset_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature cols: {missing_features}")
    leakage_features = [feature for feature in feature_cols if feature in ANALYSIS_ONLY_FEATURES]
    if leakage_features:
        raise ValueError(f"Feature cols contain analysis-only GT fields: {leakage_features}")


def open_or_create_doc(docx_path: Path) -> Document:
    if docx_path.exists():
        try:
            return Document(str(docx_path))
        except (PackageNotFoundError, BadZipFile):
            pass
    doc = Document()
    doc.add_heading(DOC_TITLE, level=1)
    return doc


def format_float(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"


def mean_of_numeric(rows: list[dict], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        raw_value = row.get(key, "")
        if raw_value in {"", None}:
            continue
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric_value):
            continue
        values.append(numeric_value)
    if not values:
        return None
    return float(sum(values) / len(values))


def normalize_optional_positive_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(numeric) or numeric <= 0.0:
        return np.nan
    return float(numeric)


def compute_count_error_pct(count_diff: object, gt_count: object) -> float:
    try:
        gt_numeric = float(gt_count)
        diff_numeric = abs(float(count_diff))
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(gt_numeric) or gt_numeric <= 0.0 or not np.isfinite(diff_numeric):
        return np.nan
    return float((diff_numeric / gt_numeric) * 100.0)


def compute_overall_count_error_pct(rows: list[dict]) -> float | None:
    total_gt = 0.0
    total_abs_diff = 0.0
    for row in rows:
        try:
            gt_numeric = float(row.get("N_gt_points", np.nan))
            diff_numeric = abs(float(row.get("count_diff", np.nan)))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(gt_numeric) or gt_numeric <= 0.0 or not np.isfinite(diff_numeric):
            continue
        total_gt += gt_numeric
        total_abs_diff += diff_numeric
    if total_gt <= 0.0:
        return None
    return float((total_abs_diff / total_gt) * 100.0)


def append_result_to_docx(docx_path: Path, title: str, config: dict, results: list[dict]) -> None:
    doc = open_or_create_doc(docx_path)
    doc.add_heading(title, level=2)
    doc.add_paragraph(f"Record time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    config_lines = [
        "task: raw-point-guided segment refine",
        f"script: {ROOT / 'raw_point_guided_segment_refine.py'}",
        f"exist_exp_dir: {config['exist_exp_dir']}",
        f"sample_dir: {config['sample_dir']}",
        f"raw_label_dir: {config['raw_label_dir']}",
        f"gt_dev_rule: {config['gt_dev_rule']}",
        f"model_name: {config['model_name']}",
        f"model_alpha: {config['model_alpha']}",
        f"model_max_iter: {config['model_max_iter']}",
        f"tweedie_power: {config['tweedie_power']}",
        f"hgb_learning_rate: {config['hgb_learning_rate']}",
        f"hgb_max_depth: {config['hgb_max_depth']}",
        f"feature_preset: {config['feature_preset']}",
        f"feature_cols: {config['feature_cols']}",
        f"prob_weight_gamma: {config['prob_weight_gamma']}",
        f"fallback_count_mode: {config['fallback_count_mode']}",
        f"count_fusion_mode: {config['count_fusion_mode']}",
        f"fusion_learned_weight: {config['fusion_learned_weight']}",
        f"global_post_scale_mode: {config['global_post_scale_mode']}",
        f"global_post_scale_min: {config['global_post_scale_min']}",
        f"global_post_scale_max: {config['global_post_scale_max']}",
        f"selective_post_mode: {config['selective_post_mode']}",
        f"selective_post_target_wells: {config['selective_post_target_wells']}",
        f"selective_post_feature: {config['selective_post_feature']}",
        f"selective_post_feature_2: {config['selective_post_feature_2']}",
        f"selective_post_quantile: {config['selective_post_quantile']}",
        f"selective_post_quantile_2: {config['selective_post_quantile_2']}",
        f"selective_post_max_pred_float: {config['selective_post_max_pred_float']}",
        f"boundary_expand_mode: {config['boundary_expand_mode']}",
        f"boundary_expand_prob_min: {config['boundary_expand_prob_min']}",
        f"boundary_expand_max_steps: {config['boundary_expand_max_steps']}",
        f"boundary_expand_max_depth: {config['boundary_expand_max_depth']}",
        f"soft_negative_weight: {config['soft_negative_weight']}",
        f"pred_min_points_per_segment: {config['pred_min_points_per_segment']}",
        f"rounding_mode: {config['rounding_mode']}",
        f"save_dir: {config['save_dir']}",
        f"summary_csv: {config['summary_csv']}",
    ]
    if config["model_name"] == "xgboost":
        config_lines.extend(
            [
                f"xgb_n_estimators: {config['xgb_n_estimators']}",
                f"xgb_max_depth: {config['xgb_max_depth']}",
                f"xgb_learning_rate: {config['xgb_learning_rate']}",
                f"xgb_subsample: {config['xgb_subsample']}",
                f"xgb_colsample_bytree: {config['xgb_colsample_bytree']}",
            ]
        )
    doc.add_paragraph("\n".join(config_lines))

    table = doc.add_table(rows=1, cols=len(DEFAULT_RESULT_HEADERS))
    for idx, header in enumerate(DEFAULT_RESULT_HEADERS):
        table.rows[0].cells[idx].text = header
    for row in results:
        cells = table.add_row().cells
        for idx, key in enumerate(DEFAULT_RESULT_KEYS):
            cells[idx].text = str(row[key]) if key == "well" else format_float(row[key])

    summary_parts = []
    for key, label in (
        ("count_diff", "AVG_count_diff"),
        ("count_error_pct", "AVG_count_error_pct"),
        ("pred_to_gt_mean_dist", "AVG_pred_to_gt_mean"),
        ("gt_to_pred_mean_dist", "AVG_gt_to_pred_mean"),
        ("segment_count_mae", "AVG_segment_count_mae"),
    ):
        avg_value = mean_of_numeric(results, key)
        if avg_value is not None:
            summary_parts.append(f"{label}={avg_value:.4f}")
    overall_count_error_pct = compute_overall_count_error_pct(results)
    if overall_count_error_pct is not None:
        summary_parts.append(f"TOTAL_count_error_pct={overall_count_error_pct:.4f}")
    doc.add_paragraph("Summary: " + ", ".join(summary_parts) if summary_parts else "Summary: recorded")
    doc.add_paragraph("")
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(docx_path))


def interval_overlap_length(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    left = max(float(start_a), float(start_b))
    right = min(float(end_a), float(end_b))
    return float(max(right - left, 0.0))


def compute_overlap_with_gt_dev(start_depth: float, end_depth: float, gt_dev_segments: pd.DataFrame) -> tuple[float, int]:
    if gt_dev_segments.empty:
        return 0.0, 0
    overlap_len = 0.0
    overlap_count = 0
    for row in gt_dev_segments.itertuples(index=False):
        current_len = interval_overlap_length(start_depth, end_depth, row.SegStartDepth, row.SegEndDepth)
        if current_len > 0:
            overlap_len += current_len
            overlap_count += 1
    return float(overlap_len), int(overlap_count)


def count_points_in_any_segments(point_depths: np.ndarray, segment_df: pd.DataFrame) -> int:
    if point_depths.size == 0 or segment_df.empty:
        return 0
    count = 0
    for depth_value in point_depths:
        in_any = (
            (segment_df["SegStartDepth"].to_numpy(dtype=np.float64) <= depth_value)
            & (segment_df["SegEndDepth"].to_numpy(dtype=np.float64) >= depth_value)
        )
        if bool(np.any(in_any)):
            count += 1
    return int(count)


def apply_boundary_expand_to_pred_mask(
    df: pd.DataFrame,
    config,
    mode: str,
    prob_min: float,
    max_steps: int,
    max_depth: float,
) -> tuple[pd.DataFrame, str, dict]:
    if config.depth_col not in df.columns:
        raise ValueError(f"Missing depth col {config.depth_col}")
    if config.pred_label_col not in df.columns:
        raise ValueError(f"Missing pred label col {config.pred_label_col}")

    normalized_mode = validate_boundary_expand_mode(mode)
    out = df.copy()
    out[config.depth_col] = dataset_builder.POINT.safe_to_numeric(out[config.depth_col])
    out = out.sort_values(config.depth_col).reset_index(drop=True)

    base_mask = (
        dataset_builder.POINT.safe_to_numeric(out[config.pred_label_col])
        .fillna(0.0)
        .astype(int)
        .to_numpy(dtype=np.int8)
        == 1
    )
    effective_mask = base_mask.copy()
    base_segments = dataset_builder.POINT.find_binary_segments(base_mask)

    clipped_prob_min = float(np.clip(prob_min, 0.0, 1.0))
    clipped_max_steps = max(int(max_steps), 0)
    clipped_max_depth = float(max_depth)
    limit_depth = np.isfinite(clipped_max_depth) and clipped_max_depth > 0.0
    added_left = 0
    added_right = 0

    if normalized_mode == "prob_shoulder" and base_segments:
        if config.pred_prob_col not in out.columns:
            raise ValueError(f"Missing pred prob col {config.pred_prob_col}")
        depth = out[config.depth_col].to_numpy(dtype=np.float64)
        prob = (
            dataset_builder.POINT.safe_to_numeric(out[config.pred_prob_col])
            .fillna(0.0)
            .clip(lower=0.0, upper=1.0)
            .to_numpy(dtype=np.float64)
        )

        for start_idx, end_idx in base_segments:
            start_depth = float(depth[start_idx])
            end_depth = float(depth[end_idx])

            current_idx = start_idx - 1
            steps_used = 0
            while current_idx >= 0 and steps_used < clipped_max_steps:
                if effective_mask[current_idx]:
                    current_idx -= 1
                    continue
                current_prob = float(prob[current_idx])
                current_depth = float(depth[current_idx])
                if not np.isfinite(current_prob) or current_prob < clipped_prob_min:
                    break
                if limit_depth and np.isfinite(current_depth) and (start_depth - current_depth) > clipped_max_depth:
                    break
                effective_mask[current_idx] = True
                added_left += 1
                steps_used += 1
                current_idx -= 1

            current_idx = end_idx + 1
            steps_used = 0
            while current_idx < len(effective_mask) and steps_used < clipped_max_steps:
                if effective_mask[current_idx]:
                    current_idx += 1
                    continue
                current_prob = float(prob[current_idx])
                current_depth = float(depth[current_idx])
                if not np.isfinite(current_prob) or current_prob < clipped_prob_min:
                    break
                if limit_depth and np.isfinite(current_depth) and (current_depth - end_depth) > clipped_max_depth:
                    break
                effective_mask[current_idx] = True
                added_right += 1
                steps_used += 1
                current_idx += 1

    out[BOUNDARY_EXPAND_EFFECTIVE_MASK_COL] = effective_mask.astype(np.int8)
    effective_segments = dataset_builder.POINT.find_binary_segments(effective_mask)
    stats = {
        "boundary_expand_mode": normalized_mode,
        "boundary_expand_prob_min": clipped_prob_min,
        "boundary_expand_max_steps": clipped_max_steps,
        "boundary_expand_max_depth": clipped_max_depth,
        "boundary_base_positive_samples": int(np.sum(base_mask.astype(np.int64))),
        "boundary_effective_positive_samples": int(np.sum(effective_mask.astype(np.int64))),
        "boundary_added_positive_samples": int(np.sum(effective_mask.astype(np.int64)) - np.sum(base_mask.astype(np.int64))),
        "boundary_base_segment_count": int(len(base_segments)),
        "boundary_effective_segment_count": int(len(effective_segments)),
        "boundary_added_left_samples": int(added_left),
        "boundary_added_right_samples": int(added_right),
    }
    return out, BOUNDARY_EXPAND_EFFECTIVE_MASK_COL, stats




def excel_ref_to_col_idx(cell_ref: str) -> int:
    col = 0
    for ch in str(cell_ref):
        if not ch.isalpha():
            break
        col = (col * 26) + (ord(ch.upper()) - ord("A") + 1)
    return max(col - 1, 0)


def load_shared_strings(xlsx_file: ZipFile) -> list[str]:
    try:
        xml_bytes = xlsx_file.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(xml_bytes)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    values = []
    for item in root.findall(f"{ns}si"):
        text_parts = []
        for text_node in item.iter(f"{ns}t"):
            text_parts.append(text_node.text or "")
        values.append("".join(text_parts))
    return values


def read_xlsx_cell_value(cell, shared_strings: list[str]) -> str:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        text_node = cell.find(f"{ns}is/{ns}t")
        return "" if text_node is None else str(text_node.text or "")
    value_node = cell.find(f"{ns}v")
    if value_node is None:
        return ""
    raw_text = "" if value_node.text is None else str(value_node.text)
    if cell_type == "s":
        try:
            idx = int(float(raw_text))
        except ValueError:
            return raw_text
        return shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
    return raw_text


def read_simple_xlsx(path: Path, names: list[str]) -> pd.DataFrame:
    with ZipFile(path) as xlsx_file:
        shared_strings = load_shared_strings(xlsx_file)
        sheet_xml = xlsx_file.read("xl/worksheets/sheet1.xml")
    root = ET.fromstring(sheet_xml)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows = []
    for row_node in root.iter(f"{ns}row"):
        current_row = {}
        for cell in row_node.findall(f"{ns}c"):
            col_idx = excel_ref_to_col_idx(cell.attrib.get("r", ""))
            current_row[col_idx] = read_xlsx_cell_value(cell, shared_strings)
        if current_row:
            width = max(current_row.keys()) + 1
            rows.append([current_row.get(idx, "") for idx in range(width)])
    if not rows:
        return pd.DataFrame(columns=names)
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    df = pd.DataFrame(normalized)
    if df.shape[1] < len(names):
        for _ in range(len(names) - df.shape[1]):
            df[df.shape[1]] = ""
    df = df.iloc[:, : len(names)].copy()
    df.columns = names
    if not df.empty:
        first_value = pd.to_numeric(df[names[0]], errors="coerce").iloc[0]
        if not np.isfinite(first_value):
            df = df.iloc[1:].reset_index(drop=True)
    return df

def load_raw_fracture_points(raw_label_dir: Path, well_name: str, depth_col: str = "TVD") -> pd.DataFrame:
    raw_file_name = WELL_TO_RAW_FILE.get(well_name)
    if not raw_file_name:
        raise KeyError(f"No raw label file mapping for well: {well_name}")
    raw_path = raw_label_dir / raw_file_name
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw label file not found for {well_name}: {raw_path}")

    raw_df = read_simple_xlsx(raw_path, ["MD", "Angle(0~90)", "Azimuth(0~360)"])
    raw_df = raw_df.rename(
        columns={
            "Angle(0~90)": "Frac_Dip",
            "Azimuth(0~360)": "Frac_Azimuth",
        }
    )
    raw_df["MD"] = dataset_builder.POINT.safe_to_numeric(raw_df["MD"])
    raw_df["Frac_Dip"] = dataset_builder.POINT.safe_to_numeric(raw_df["Frac_Dip"])
    raw_df["Frac_Azimuth"] = dataset_builder.POINT.safe_to_numeric(raw_df["Frac_Azimuth"])
    raw_df = raw_df.dropna(subset=["MD"]).copy()
    raw_df = raw_df.sort_values("MD").drop_duplicates(subset=["MD"], keep="first").reset_index(drop=True)
    raw_df[depth_col] = raw_df["MD"].astype(np.float64)
    raw_df["RawPointID"] = np.arange(1, len(raw_df) + 1, dtype=np.int64)
    raw_df["WellName"] = well_name
    raw_df["SourceFile"] = str(raw_path)
    return raw_df[["WellName", "RawPointID", "MD", depth_col, "Frac_Dip", "Frac_Azimuth", "SourceFile"]]


def build_gt_dev_segments(sample_dir: Path, well_name: str, gt_dev_rule: str, depth_col: str = "TVD") -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_file_name = WELL_TO_SAMPLE_FILE.get(well_name)
    if not sample_file_name:
        raise KeyError(f"No sample file mapping for well: {well_name}")
    sample_path = sample_dir / sample_file_name
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found for {well_name}: {sample_path}")

    sample_df = pd.read_csv(sample_path, encoding="utf-8-sig")
    if depth_col not in sample_df.columns:
        raise ValueError(f"Missing depth col {depth_col} in sample file: {sample_path}")

    sample_df = sample_df.copy()
    sample_df[depth_col] = dataset_builder.POINT.safe_to_numeric(sample_df[depth_col])
    for col in ["P10", "P21", "P33"]:
        if col in sample_df.columns:
            sample_df[col] = dataset_builder.POINT.safe_to_numeric(sample_df[col]).fillna(0.0)
    if "Frac_Azimuth" in sample_df.columns:
        sample_df["Frac_Azimuth"] = dataset_builder.POINT.safe_to_numeric(sample_df["Frac_Azimuth"])
    if "GT_LABEL" in sample_df.columns:
        sample_df["GT_LABEL"] = dataset_builder.POINT.safe_to_numeric(sample_df["GT_LABEL"]).fillna(0.0)

    sample_df = sample_df.dropna(subset=[depth_col]).sort_values(depth_col).reset_index(drop=True)
    if gt_dev_rule == "any_density":
        mask = (
            sample_df[[col for col in ["P10", "P21", "P33"] if col in sample_df.columns]]
            .fillna(0.0)
            .max(axis=1)
            .gt(0.0)
        )
    elif gt_dev_rule == "p10_only":
        if "P10" not in sample_df.columns:
            raise ValueError(f"P10 missing in sample file: {sample_path}")
        mask = sample_df["P10"].fillna(0.0).gt(0.0)
    elif gt_dev_rule == "frac_azimuth_notna":
        if "Frac_Azimuth" not in sample_df.columns:
            raise ValueError(f"Frac_Azimuth missing in sample file: {sample_path}")
        mask = sample_df["Frac_Azimuth"].notna()
    elif gt_dev_rule == "gt_label":
        if "GT_LABEL" not in sample_df.columns:
            raise ValueError(f"GT_LABEL missing in sample file: {sample_path}")
        mask = sample_df["GT_LABEL"].fillna(0.0).ge(1.0)
    else:
        raise ValueError(f"Unsupported gt_dev_rule: {gt_dev_rule}")

    depth = sample_df[depth_col].to_numpy(dtype=np.float64)
    segments = dataset_builder.POINT.find_binary_segments(mask.to_numpy(dtype=bool))
    gt_rows = []
    for seg_id, (start_idx, end_idx) in enumerate(segments, start=1):
        depth_seg = depth[start_idx : end_idx + 1]
        seg_length = float(max(depth_seg[-1] - depth_seg[0], 0.0)) if depth_seg.size >= 2 else 0.0
        gt_rows.append(
            {
                "WellName": well_name,
                "GTDevSegment_ID": seg_id,
                "SegStartDepth": float(depth_seg[0]),
                "SegEndDepth": float(depth_seg[-1]),
                "SegLength": seg_length,
                "SampleCount": int(len(depth_seg)),
                "SourceFile": str(sample_path),
                "GTDevRule": gt_dev_rule,
            }
        )
    gt_dev_df = pd.DataFrame(gt_rows)
    return sample_df, gt_dev_df


def build_well_segment_dataset(
    exist_csv: str,
    well_name: str,
    config,
    sample_dir: Path,
    raw_label_dir: Path,
    gt_dev_rule: str,
    boundary_expand_mode: str = "none",
    boundary_expand_prob_min: float = 0.0,
    boundary_expand_max_steps: int = 0,
    boundary_expand_max_depth: float = 0.0,
) -> dict:
    df = pd.read_csv(exist_csv, encoding="utf-8-sig")
    raw_gt_points = load_raw_fracture_points(raw_label_dir=raw_label_dir, well_name=well_name, depth_col=config.depth_col)
    raw_gt_depths = (
        raw_gt_points[config.depth_col].to_numpy(dtype=np.float64)
        if not raw_gt_points.empty
        else np.array([], dtype=np.float64)
    )
    _, gt_dev_segments = build_gt_dev_segments(
        sample_dir=sample_dir,
        well_name=well_name,
        gt_dev_rule=gt_dev_rule,
        depth_col=config.depth_col,
    )
    df_for_segment, effective_mask_col, boundary_stats = apply_boundary_expand_to_pred_mask(
        df=df,
        config=config,
        mode=boundary_expand_mode,
        prob_min=boundary_expand_prob_min,
        max_steps=boundary_expand_max_steps,
        max_depth=boundary_expand_max_depth,
    )
    df_sorted, payloads = dataset_builder.POINT.extract_segment_payloads(
        df_for_segment,
        mask_col=effective_mask_col,
        density_col=None,
        config=config,
        prob_col=config.pred_prob_col,
        shape_mode="prob_only",
    )

    rows = []
    for payload in payloads:
        row = dataset_builder.build_segment_row(
            well_name=well_name,
            payload=payload,
            df_sorted=df_sorted,
            gt_depths=raw_gt_depths,
            config=config,
        )
        raw_gt_count = int(row.get("GTPointCountInPredSegment", 0))
        overlap_len, overlap_count = compute_overlap_with_gt_dev(
            start_depth=float(payload["SegStartDepth"]),
            end_depth=float(payload["SegEndDepth"]),
            gt_dev_segments=gt_dev_segments,
        )
        seg_length = float(max(row.get("SegLength", 0.0), 0.0))
        row.update(
            {
                "GTPointCountInPredSegment": raw_gt_count,
                "RawGTCountInPredSegment": raw_gt_count,
                "HasRawPoint": int(raw_gt_count > 0),
                "GTDevOverlapLen": float(overlap_len),
                "GTDevOverlapRatio": float(overlap_len / seg_length) if seg_length > 1e-8 else 0.0,
                "GTDevOverlapCount": int(overlap_count),
                "GTDevOverlapFlag": int(overlap_len > 0),
            }
        )
        rows.append(row)

    segment_df = pd.DataFrame(rows)
    raw_points_covered = count_points_in_any_segments(raw_gt_depths, segment_df)
    return {
        "well_name": well_name,
        "exist_csv": exist_csv,
        "df": df,
        "df_effective": df_for_segment,
        "df_sorted": df_sorted,
        "payloads": payloads,
        "raw_gt_points": raw_gt_points,
        "raw_gt_depths": raw_gt_depths,
        "gt_dev_segments": gt_dev_segments,
        "segment_df": segment_df,
        "boundary_expand_stats": boundary_stats,
        "raw_points_covered_by_pred_segments": int(raw_points_covered),
        "raw_points_missed_outside_pred_segments": int(max(len(raw_gt_depths) - raw_points_covered, 0)),
    }


def build_predict_segment_dataset(
    exist_csv: str,
    well_name: str,
    config,
    boundary_expand_mode: str = "none",
    boundary_expand_prob_min: float = 0.0,
    boundary_expand_max_steps: int = 0,
    boundary_expand_max_depth: float = 0.0,
) -> dict:
    df = pd.read_csv(exist_csv, encoding="utf-8-sig")
    df_for_segment, effective_mask_col, boundary_stats = apply_boundary_expand_to_pred_mask(
        df=df,
        config=config,
        mode=boundary_expand_mode,
        prob_min=boundary_expand_prob_min,
        max_steps=boundary_expand_max_steps,
        max_depth=boundary_expand_max_depth,
    )
    df_sorted, payloads = dataset_builder.POINT.extract_segment_payloads(
        df_for_segment,
        mask_col=effective_mask_col,
        density_col=None,
        config=config,
        prob_col=config.pred_prob_col,
        shape_mode="prob_only",
    )

    empty_gt_depths = np.array([], dtype=np.float64)
    rows = []
    for payload in payloads:
        row = dataset_builder.build_segment_row(
            well_name=well_name,
            payload=payload,
            df_sorted=df_sorted,
            gt_depths=empty_gt_depths,
            config=config,
        )
        row.update(
            {
                "RawGTCountInPredSegment": 0,
                "HasRawPoint": 0,
                "GTDevOverlapLen": 0.0,
                "GTDevOverlapRatio": 0.0,
                "GTDevOverlapCount": 0,
                "GTDevOverlapFlag": 0,
                "TargetWeight": np.nan,
                "TargetLabelType": "",
            }
        )
        rows.append(row)

    return {
        "well_name": well_name,
        "exist_csv": exist_csv,
        "df": df,
        "df_effective": df_for_segment,
        "df_sorted": df_sorted,
        "payloads": payloads,
        "boundary_expand_stats": boundary_stats,
        "segment_df": pd.DataFrame(rows),
    }


def compute_target_weights(segment_df: pd.DataFrame, soft_negative_weight: float) -> pd.DataFrame:
    if segment_df.empty:
        out = segment_df.copy()
        out["TargetWeight"] = []
        out["TargetLabelType"] = []
        return out

    clipped_soft_weight = float(np.clip(soft_negative_weight, 0.0, 1.0))
    target_weights = []
    target_types = []
    for row in segment_df.itertuples(index=False):
        raw_count = int(getattr(row, "RawGTCountInPredSegment", 0))
        overlap_len = float(getattr(row, "GTDevOverlapLen", 0.0))
        if raw_count > 0:
            target_weights.append(1.0)
            target_types.append("hard_positive")
        elif overlap_len > 0:
            target_weights.append(clipped_soft_weight)
            target_types.append("soft_zero_overlap_gt_dev")
        else:
            target_weights.append(1.0)
            target_types.append("hard_zero")

    out = segment_df.copy()
    out["TargetWeight"] = np.asarray(target_weights, dtype=np.float64)
    out["TargetLabelType"] = target_types
    return out


def fit_basis_scale(train_df: pd.DataFrame, count_mode: str) -> tuple[str, float]:
    basis_col = dataset_builder.POINT.COUNT_MODE_TO_BASIS_COL[count_mode]
    basis_sum = float(train_df[basis_col].fillna(0.0).clip(lower=0.0).sum())
    target_sum = float(train_df["RawGTCountInPredSegment"].fillna(0.0).clip(lower=0.0).sum())
    scale = (target_sum / basis_sum) if basis_sum > 1e-8 else 1.0
    return basis_col, float(max(scale, 0.0))


def fuse_count_predictions(
    learned_pred: np.ndarray,
    fallback_pred: np.ndarray | None,
    fusion_mode: str,
    learned_weight: float,
) -> np.ndarray:
    learned_pred = np.clip(np.asarray(learned_pred, dtype=np.float64), a_min=0.0, a_max=None)
    if fallback_pred is None or not fusion_mode or fusion_mode == "learned_only":
        return learned_pred

    fallback_pred = np.clip(np.asarray(fallback_pred, dtype=np.float64), a_min=0.0, a_max=None)
    if fusion_mode == "max":
        return np.maximum(learned_pred, fallback_pred)
    if fusion_mode == "blend":
        weight = float(np.clip(learned_weight, 0.0, 1.0))
        return (weight * learned_pred) + ((1.0 - weight) * fallback_pred)
    if fusion_mode == "uplift":
        weight = float(np.clip(learned_weight, 0.0, 1.0))
        positive_gap = np.clip(fallback_pred - learned_pred, a_min=0.0, a_max=None)
        return learned_pred + ((1.0 - weight) * positive_gap)
    raise ValueError(f"Unsupported count_fusion_mode: {fusion_mode}")


def fit_global_post_scale(
    train_target: np.ndarray,
    train_pred: np.ndarray,
    mode: str,
    scale_min: float,
    scale_max: float,
) -> dict:
    normalized_mode = validate_global_post_scale_mode(mode)
    lower = float(scale_min)
    upper = float(scale_max)
    if upper < lower:
        raise ValueError(f"global_post_scale_max must be >= global_post_scale_min, got {upper} < {lower}")

    target = np.clip(np.asarray(train_target, dtype=np.float64), a_min=0.0, a_max=None)
    pred = np.clip(np.asarray(train_pred, dtype=np.float64), a_min=0.0, a_max=None)
    train_target_sum = float(np.sum(target))
    train_pred_sum = float(np.sum(pred))
    raw_scale = (train_target_sum / train_pred_sum) if train_pred_sum > 1e-8 else 1.0

    if normalized_mode == "none":
        applied_scale = 1.0
    elif normalized_mode == "train_sum_match":
        applied_scale = raw_scale
    elif normalized_mode == "train_sum_match_up_only":
        applied_scale = max(raw_scale, 1.0)
    else:
        raise ValueError(f"Unsupported global_post_scale_mode: {normalized_mode}")

    applied_scale = float(np.clip(applied_scale, lower, upper))
    return {
        "mode": normalized_mode,
        "raw_scale": float(raw_scale),
        "scale": applied_scale,
        "train_target_sum": train_target_sum,
        "train_pred_sum": train_pred_sum,
        "scale_min": lower,
        "scale_max": upper,
    }


def apply_global_post_scale(pred_count_float: np.ndarray, post_scale_info: dict) -> np.ndarray:
    scale = float(post_scale_info.get("scale", 1.0))
    pred = np.asarray(pred_count_float, dtype=np.float64)
    return np.clip(pred * scale, a_min=0.0, a_max=None)


def fit_selective_post_strategy(
    train_df: pd.DataFrame,
    train_pred_after_global_scale: np.ndarray,
    mode: str,
    feature_name: str,
    feature_name_2: str,
    quantile: float,
    quantile_2: float,
    fallback_scale: float,
    max_pred_float: float,
) -> dict:
    normalized_mode = validate_selective_post_mode(mode)
    clipped_quantile = float(np.clip(float(quantile), 0.0, 1.0))
    clipped_quantile_2 = float(np.clip(float(quantile_2), 0.0, 1.0))
    clipped_max_pred_float = normalize_optional_positive_float(max_pred_float)
    feature_values = train_df.get(feature_name)
    feature_values_2 = train_df.get(feature_name_2) if feature_name_2 else None
    if normalized_mode == "none":
        return {
            "mode": normalized_mode,
            "feature_name": feature_name,
            "feature_name_2": feature_name_2,
            "quantile": clipped_quantile,
            "quantile_2": clipped_quantile_2,
            "threshold": np.nan,
            "threshold_2": np.nan,
            "lowconf_scale": 1.0,
            "max_pred_float": clipped_max_pred_float,
            "num_train_zero_segments": 0,
        }

    if feature_values is None:
        raise ValueError(f"Missing selective post feature: {feature_name}")
    if normalized_mode == "train_zero_dual_quantile_downscale" and feature_values_2 is None:
        raise ValueError(f"Missing selective post feature 2: {feature_name_2}")

    zero_mask = train_df["RawGTCountInPredSegment"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64) <= 0.0
    zero_feature_values = (
        pd.to_numeric(feature_values, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    zero_feature_values = zero_feature_values[zero_mask]
    zero_feature_values = zero_feature_values[np.isfinite(zero_feature_values)]
    if zero_feature_values.size == 0:
        threshold = np.nan
    else:
        threshold = float(np.quantile(zero_feature_values, clipped_quantile))

    threshold_2 = np.nan
    if normalized_mode == "train_zero_dual_quantile_downscale":
        zero_feature_values_2 = (
            pd.to_numeric(feature_values_2, errors="coerce")
            .to_numpy(dtype=np.float64)
        )
        zero_feature_values_2 = zero_feature_values_2[zero_mask]
        zero_feature_values_2 = zero_feature_values_2[np.isfinite(zero_feature_values_2)]
        if zero_feature_values_2.size > 0:
            threshold_2 = float(np.quantile(zero_feature_values_2, clipped_quantile_2))

    lowconf_scale = float(np.clip(fallback_scale, 0.0, 1.0))
    return {
        "mode": normalized_mode,
        "feature_name": feature_name,
        "feature_name_2": feature_name_2,
        "quantile": clipped_quantile,
        "quantile_2": clipped_quantile_2,
        "threshold": threshold,
        "threshold_2": threshold_2,
        "lowconf_scale": lowconf_scale,
        "max_pred_float": clipped_max_pred_float,
        "num_train_zero_segments": int(zero_feature_values.size),
    }


def apply_selective_post_strategy(
    pred_count_float: np.ndarray,
    val_df: pd.DataFrame,
    strategy: dict,
    pred_count_float_for_protection: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.clip(np.asarray(pred_count_float, dtype=np.float64), a_min=0.0, a_max=None)
    lowconf_mask = np.zeros(pred.shape[0], dtype=bool)
    if strategy.get("mode") not in {"train_zero_quantile_downscale", "train_zero_dual_quantile_downscale"}:
        return pred, lowconf_mask

    threshold = strategy.get("threshold")
    feature_name = str(strategy.get("feature_name"))
    if feature_name not in val_df.columns or not np.isfinite(threshold):
        return pred, lowconf_mask

    feature_values = pd.to_numeric(val_df[feature_name], errors="coerce").to_numpy(dtype=np.float64)
    lowconf_mask = np.isfinite(feature_values) & (feature_values <= float(threshold))
    if strategy.get("mode") == "train_zero_dual_quantile_downscale":
        threshold_2 = strategy.get("threshold_2")
        feature_name_2 = str(strategy.get("feature_name_2"))
        if feature_name_2 not in val_df.columns or not np.isfinite(threshold_2):
            return pred, np.zeros(pred.shape[0], dtype=bool)
        feature_values_2 = pd.to_numeric(val_df[feature_name_2], errors="coerce").to_numpy(dtype=np.float64)
        lowconf_mask = lowconf_mask & np.isfinite(feature_values_2) & (feature_values_2 <= float(threshold_2))
    max_pred_float = normalize_optional_positive_float(strategy.get("max_pred_float", np.nan))
    if np.isfinite(max_pred_float):
        protect_values = pred if pred_count_float_for_protection is None else np.asarray(
            pred_count_float_for_protection,
            dtype=np.float64,
        )
        lowconf_mask = lowconf_mask & np.isfinite(protect_values) & (protect_values <= max_pred_float)
    lowconf_scale = float(strategy.get("lowconf_scale", 1.0))
    if lowconf_scale >= 0.999999:
        return pred, lowconf_mask

    out = pred.copy()
    out[lowconf_mask] = out[lowconf_mask] * lowconf_scale
    return np.clip(out, a_min=0.0, a_max=None), lowconf_mask


def build_regressor(
    model_name: str,
    alpha: float,
    max_iter: int,
    tweedie_power: float,
    xgb_n_estimators: int,
    xgb_max_depth: int,
    xgb_learning_rate: float,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
    hgb_learning_rate: float,
    hgb_max_depth: int,
):
    if model_name == "poisson":
        model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    if model_name == "tweedie":
        model = TweedieRegressor(power=tweedie_power, alpha=alpha, max_iter=max_iter, link="log")
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    if model_name == "hgbt_poisson":
        model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=hgb_learning_rate,
            max_iter=max_iter,
            max_depth=hgb_max_depth,
            min_samples_leaf=5,
            l2_regularization=max(float(alpha), 0.0),
            random_state=42,
        )
        return Pipeline([("model", model)])
    if model_name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed in current environment")
        model = XGBRegressor(
            objective="count:poisson",
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            reg_lambda=max(float(alpha), 0.0),
            random_state=42,
        )
        return Pipeline([("model", model)])
    raise ValueError(f"Unsupported model_name: {model_name}")


def fit_regressor(regressor: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, sample_weight: np.ndarray | None):
    if sample_weight is None:
        regressor.fit(x_train, y_train)
    else:
        regressor.fit(x_train, y_train, model__sample_weight=sample_weight)
    return regressor


def compute_pred_count(raw_value: float, pred_min_points_per_segment: int, rounding_mode: str) -> int:
    return int(
        dataset_builder.POINT.compute_n_points(
            float(max(raw_value, 0.0)),
            int(pred_min_points_per_segment),
            rounding_mode,
        )
    )


def build_pred_points_from_segment_df(
    df_sorted: pd.DataFrame,
    payloads: list[dict],
    pred_segment_df: pd.DataFrame,
    config,
) -> pd.DataFrame:
    pred_map = pred_segment_df.set_index("Segment_ID").to_dict("index")
    rows = []
    for payload in payloads:
        segment_id = int(payload["Segment_ID"])
        pred_info = pred_map.get(segment_id)
        if pred_info is None:
            continue
        n_points = int(pred_info["PredPointCount"])
        if n_points <= 0:
            continue
        picked_depths, picked_rel_idx = dataset_builder.POINT.pick_points_by_equal_intensity(
            payload["depth_seg"],
            payload["shape_intensity"],
            n_points,
        )
        for point_idx, (depth_value, rel_idx) in enumerate(zip(picked_depths, picked_rel_idx), start=1):
            abs_idx = int(payload["StartIdx"] + rel_idx)
            row = df_sorted.iloc[abs_idx]
            rows.append(
                {
                    "Segment_ID": segment_id,
                    "Point_ID_In_Segment": point_idx,
                    config.depth_col: float(depth_value),
                    "NearestSampleDepth": float(row[config.depth_col]),
                    config.x_col: float(row[config.x_col]) if config.x_col in df_sorted.columns else np.nan,
                    config.y_col: float(row[config.y_col]) if config.y_col in df_sorted.columns else np.nan,
                    config.time_col: float(row[config.time_col]) if config.time_col in df_sorted.columns else np.nan,
                    "SegStartDepth": float(pred_info["SegStartDepth"]),
                    "SegEndDepth": float(pred_info["SegEndDepth"]),
                    "SegLength": float(pred_info["SegLength"]),
                    "PredPointCountFloat": float(pred_info["PredPointCountFloat"]),
                    "PredPointCount": n_points,
                    "RawGTCountInPredSegment": int(pred_info["RawGTCountInPredSegment"]),
                    "PredProbMean": float(pred_info["ProbMean"]),
                    "PredProbMax": float(pred_info["ProbMax"]),
                }
            )
    return pd.DataFrame(rows)


def build_coef_rows(well_name: str, regressor: Pipeline, feature_cols: list[str]) -> list[dict]:
    model = regressor.named_steps["model"]
    rows = []
    if hasattr(model, "coef_"):
        for feature_name, coef in zip(feature_cols, model.coef_):
            rows.append({"val_well": well_name, "feature": feature_name, "coef": float(coef)})
    if hasattr(model, "intercept_"):
        rows.append({"val_well": well_name, "feature": "__intercept__", "coef": float(model.intercept_)})
    elif hasattr(model, "feature_importances_"):
        for feature_name, importance in zip(feature_cols, model.feature_importances_):
            rows.append({"val_well": well_name, "feature": feature_name, "coef": float(importance)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--exist-exp-dir", default=str(DEFAULT_EXIST_EXP_DIR))
    parser.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--raw-label-dir", default=str(DEFAULT_RAW_LABEL_DIR))
    parser.add_argument("--docx-path", default=str(DEFAULT_DOCX_PATH))
    parser.add_argument("--result-dir")
    parser.add_argument("--well-names", default="")
    parser.add_argument("--gt-dev-rule", default="any_density")
    parser.add_argument("--prob-weight-gamma", type=float, default=2.0)
    parser.add_argument("--pred-min-points-per-segment", type=int, default=0)
    parser.add_argument("--rounding-mode", default="round")
    parser.add_argument("--fallback-count-mode", default="")
    parser.add_argument("--count-fusion-mode", default="learned_only")
    parser.add_argument("--fusion-learned-weight", type=float, default=0.7)
    parser.add_argument("--global-post-scale-mode", default="none")
    parser.add_argument("--global-post-scale-min", type=float, default=0.5)
    parser.add_argument("--global-post-scale-max", type=float, default=1.5)
    parser.add_argument("--selective-post-mode", default="none")
    parser.add_argument("--selective-post-target-wells", default="")
    parser.add_argument("--selective-post-feature", default="ProbMassPerLength")
    parser.add_argument("--selective-post-feature-2", default="ProbMean")
    parser.add_argument("--selective-post-quantile", type=float, default=0.5)
    parser.add_argument("--selective-post-quantile-2", type=float, default=0.5)
    parser.add_argument("--soft-negative-weight", type=float, default=0.5)
    parser.add_argument("--model-name", default="tweedie")
    parser.add_argument("--model-alpha", type=float, default=0.03)
    parser.add_argument("--model-max-iter", type=int, default=3000)
    parser.add_argument("--tweedie-power", type=float, default=1.5)
    parser.add_argument("--feature-preset", default="high_density_compact_v1")
    parser.add_argument("--feature-cols", default=",".join(dataset_builder.DEFAULT_FEATURE_COLS))
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--hgb-max-depth", type=int, default=3)
    args = parser.parse_args()

    save_dir = Path(args.result_dir) if args.result_dir else (BASE_SAVE_DIR / sanitize(args.exp_id))
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = save_dir / "raw_point_guided_refine_summary.csv"

    verify_logs = dataset_builder.POINT.list_verify_logs(str(args.exist_exp_dir))
    selected_wells = parse_list_arg(args.well_names) if args.well_names else sorted(verify_logs.keys())
    missing_wells = [well for well in selected_wells if well not in verify_logs]
    if missing_wells:
        raise ValueError(f"Wells not found in verify logs: {missing_wells}")

    feature_cols = resolve_feature_cols(str(args.feature_preset), str(args.feature_cols))
    fallback_count_mode = validate_optional_count_mode(str(args.fallback_count_mode))
    global_post_scale_mode = validate_global_post_scale_mode(str(args.global_post_scale_mode))
    selective_post_mode = validate_selective_post_mode(str(args.selective_post_mode))
    selective_post_target_wells = set(parse_list_arg(args.selective_post_target_wells))
    config = dataset_builder.make_config(
        exist_exp_dir=str(args.exist_exp_dir),
        prob_weight_gamma=float(args.prob_weight_gamma),
        gt_min_points_per_segment=1,
        gt_rounding_mode="round",
        density_gt_col="P10",
    )

    per_well = {}
    all_rows = []
    dataset_save_dir = save_dir / "segment_dataset"
    dataset_save_dir.mkdir(parents=True, exist_ok=True)
    for well_name in selected_wells:
        well_data = build_well_segment_dataset(
            exist_csv=verify_logs[well_name],
            well_name=well_name,
            config=config,
            sample_dir=Path(args.sample_dir),
            raw_label_dir=Path(args.raw_label_dir),
            gt_dev_rule=str(args.gt_dev_rule),
        )
        well_data["segment_df"] = compute_target_weights(
            well_data["segment_df"],
            soft_negative_weight=float(args.soft_negative_weight),
        )
        per_well[well_name] = well_data
        well_dir = dataset_save_dir / well_name
        well_dir.mkdir(parents=True, exist_ok=True)
        well_data["segment_df"].to_csv(well_dir / "segment_dataset.csv", index=False, encoding="utf-8-sig")
        well_data["raw_gt_points"].to_csv(well_dir / "raw_gt_points.csv", index=False, encoding="utf-8-sig")
        well_data["gt_dev_segments"].to_csv(well_dir / "gt_dev_segments.csv", index=False, encoding="utf-8-sig")
        if not well_data["segment_df"].empty:
            all_rows.append(well_data["segment_df"])

    dataset_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if dataset_df.empty:
        raise ValueError("Segment dataset is empty")
    dataset_df.to_csv(dataset_save_dir / "all_segment_dataset.csv", index=False, encoding="utf-8-sig")
    validate_feature_cols(feature_cols, dataset_df)

    summary_rows = []
    coef_rows = []
    for well_name in selected_wells:
        train_df = dataset_df[dataset_df["WellName"] != well_name].copy()
        val_df = dataset_df[dataset_df["WellName"] == well_name].copy()
        if train_df.empty or val_df.empty:
            continue

        x_train = train_df[feature_cols].fillna(0.0)
        y_train = train_df["RawGTCountInPredSegment"].fillna(0.0).clip(lower=0.0)
        x_val = val_df[feature_cols].fillna(0.0)
        sample_weight = train_df["TargetWeight"].fillna(1.0).to_numpy(dtype=np.float64)

        learned_pred_count_float = np.zeros(len(val_df), dtype=np.float64)
        learned_train_pred_count_float = np.zeros(len(train_df), dtype=np.float64)
        regressor = None
        if str(args.model_name) != "rule_only":
            regressor = build_regressor(
                model_name=str(args.model_name),
                alpha=float(args.model_alpha),
                max_iter=int(args.model_max_iter),
                tweedie_power=float(args.tweedie_power),
                xgb_n_estimators=int(args.xgb_n_estimators),
                xgb_max_depth=int(args.xgb_max_depth),
                xgb_learning_rate=float(args.xgb_learning_rate),
                xgb_subsample=float(args.xgb_subsample),
                xgb_colsample_bytree=float(args.xgb_colsample_bytree),
                hgb_learning_rate=float(args.hgb_learning_rate),
                hgb_max_depth=int(args.hgb_max_depth),
            )
            regressor = fit_regressor(regressor, x_train, y_train, sample_weight)
            learned_train_pred_count_float = np.clip(regressor.predict(x_train), a_min=0.0, a_max=None)
            learned_pred_count_float = np.clip(regressor.predict(x_val), a_min=0.0, a_max=None)
            coef_rows.extend(build_coef_rows(well_name, regressor, feature_cols))

        fallback_basis_col = ""
        fallback_scale = np.nan
        fallback_pred_count_float = None
        fallback_train_pred_count_float = None
        if fallback_count_mode:
            fallback_basis_col, fallback_scale = fit_basis_scale(train_df, fallback_count_mode)
            fallback_train_pred_count_float = np.clip(
                train_df[fallback_basis_col].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64) * fallback_scale,
                a_min=0.0,
                a_max=None,
            )
            fallback_pred_count_float = np.clip(
                val_df[fallback_basis_col].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64) * fallback_scale,
                a_min=0.0,
                a_max=None,
            )

        if str(args.model_name) == "rule_only":
            if fallback_pred_count_float is None:
                raise ValueError("model_name=rule_only requires fallback_count_mode")
            train_pred_count_float_before_post_scale = fallback_train_pred_count_float
            pred_count_float_before_post_scale = fallback_pred_count_float
        else:
            train_pred_count_float_before_post_scale = fuse_count_predictions(
                learned_pred=learned_train_pred_count_float,
                fallback_pred=fallback_train_pred_count_float,
                fusion_mode=str(args.count_fusion_mode),
                learned_weight=float(args.fusion_learned_weight),
            )
            pred_count_float_before_post_scale = fuse_count_predictions(
                learned_pred=learned_pred_count_float,
                fallback_pred=fallback_pred_count_float,
                fusion_mode=str(args.count_fusion_mode),
                learned_weight=float(args.fusion_learned_weight),
            )

        post_scale_info = fit_global_post_scale(
            train_target=train_df["RawGTCountInPredSegment"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64),
            train_pred=train_pred_count_float_before_post_scale,
            mode=global_post_scale_mode,
            scale_min=float(args.global_post_scale_min),
            scale_max=float(args.global_post_scale_max),
        )
        train_pred_count_float_after_global_scale = apply_global_post_scale(
            train_pred_count_float_before_post_scale,
            post_scale_info,
        )
        pred_count_float_after_global_scale = apply_global_post_scale(
            pred_count_float_before_post_scale,
            post_scale_info,
        )
        selective_post_mode_this_well = (
            selective_post_mode
            if (not selective_post_target_wells or well_name in selective_post_target_wells)
            else "none"
        )
        selective_post_strategy = fit_selective_post_strategy(
            train_df=train_df,
            train_pred_after_global_scale=train_pred_count_float_after_global_scale,
            mode=selective_post_mode_this_well,
            feature_name=str(args.selective_post_feature),
            feature_name_2=str(args.selective_post_feature_2),
            quantile=float(args.selective_post_quantile),
            quantile_2=float(args.selective_post_quantile_2),
            fallback_scale=float(post_scale_info["raw_scale"]),
        )
        pred_count_float, lowconf_mask = apply_selective_post_strategy(
            pred_count_float_after_global_scale,
            val_df,
            selective_post_strategy,
        )

        pred_segment_df = val_df.copy()
        pred_segment_df["LearnedPredPointCountFloat"] = learned_pred_count_float
        pred_segment_df["FallbackCountBasisCol"] = fallback_basis_col
        pred_segment_df["FallbackCountScale"] = fallback_scale
        pred_segment_df["FallbackPredPointCountFloat"] = (
            fallback_pred_count_float if fallback_pred_count_float is not None else np.nan
        )
        pred_segment_df["PredPointCountFloatBeforePostScale"] = pred_count_float_before_post_scale
        pred_segment_df["GlobalPostScaleMode"] = post_scale_info["mode"]
        pred_segment_df["GlobalPostScaleRawScale"] = post_scale_info["raw_scale"]
        pred_segment_df["GlobalPostScaleAppliedScale"] = post_scale_info["scale"]
        pred_segment_df["GlobalPostScaleTrainTargetSum"] = post_scale_info["train_target_sum"]
        pred_segment_df["GlobalPostScaleTrainPredSum"] = post_scale_info["train_pred_sum"]
        pred_segment_df["SelectivePostMode"] = selective_post_strategy["mode"]
        pred_segment_df["SelectivePostFeature"] = selective_post_strategy["feature_name"]
        pred_segment_df["SelectivePostFeature2"] = selective_post_strategy["feature_name_2"]
        pred_segment_df["SelectivePostThreshold"] = selective_post_strategy["threshold"]
        pred_segment_df["SelectivePostThreshold2"] = selective_post_strategy["threshold_2"]
        pred_segment_df["SelectivePostQuantile"] = selective_post_strategy["quantile"]
        pred_segment_df["SelectivePostQuantile2"] = selective_post_strategy["quantile_2"]
        pred_segment_df["SelectivePostLowconfScale"] = selective_post_strategy["lowconf_scale"]
        pred_segment_df["SelectivePostIsLowConf"] = lowconf_mask.astype(int)
        pred_segment_df["PredPointCountFloat"] = pred_count_float
        pred_segment_df["PredPointCount"] = pred_segment_df["PredPointCountFloat"].apply(
            lambda value: compute_pred_count(
                raw_value=float(value),
                pred_min_points_per_segment=int(args.pred_min_points_per_segment),
                rounding_mode=str(args.rounding_mode),
            )
        )
        pred_segment_df["SegmentCountAbsError"] = (
            pred_segment_df["PredPointCount"] - pred_segment_df["RawGTCountInPredSegment"]
        ).abs()

        well_data = per_well[well_name]
        pred_points = build_pred_points_from_segment_df(
            df_sorted=well_data["df_sorted"],
            payloads=well_data["payloads"],
            pred_segment_df=pred_segment_df,
            config=config,
        )
        raw_gt_points = well_data["raw_gt_points"]
        raw_gt_depths = well_data["raw_gt_depths"]
        pred_depths = (
            pred_points[config.depth_col].to_numpy(dtype=np.float64)
            if not pred_points.empty
            else np.array([], dtype=np.float64)
        )
        pred_to_gt = dataset_builder.POINT.nearest_distance_stats(pred_depths, raw_gt_depths)
        gt_to_pred = dataset_builder.POINT.nearest_distance_stats(raw_gt_depths, pred_depths)

        well_save_dir = save_dir / well_name
        well_save_dir.mkdir(parents=True, exist_ok=True)
        raw_gt_points.to_csv(well_save_dir / "raw_gt_points.csv", index=False, encoding="utf-8-sig")
        well_data["gt_dev_segments"].to_csv(well_save_dir / "gt_dev_segments.csv", index=False, encoding="utf-8-sig")
        pred_points.to_csv(well_save_dir / "pred_fracture_points.csv", index=False, encoding="utf-8-sig")
        pred_segment_df.to_csv(well_save_dir / "pred_segment_summary.csv", index=False, encoding="utf-8-sig")
        train_df.to_csv(well_save_dir / "train_segment_dataset.csv", index=False, encoding="utf-8-sig")

        metrics = {
            "well": well_name,
            "exist_csv": well_data["exist_csv"],
            "model_name": args.model_name,
            "model_alpha": args.model_alpha,
            "model_max_iter": args.model_max_iter,
            "tweedie_power": args.tweedie_power,
            "hgb_learning_rate": args.hgb_learning_rate,
            "hgb_max_depth": args.hgb_max_depth,
            "feature_preset": args.feature_preset,
            "feature_cols": feature_cols,
            "prob_weight_gamma": args.prob_weight_gamma,
            "fallback_count_mode": fallback_count_mode,
            "count_fusion_mode": args.count_fusion_mode,
            "fusion_learned_weight": args.fusion_learned_weight,
            "global_post_scale_mode": global_post_scale_mode,
            "global_post_scale_raw_scale": post_scale_info["raw_scale"],
            "global_post_scale_applied_scale": post_scale_info["scale"],
            "global_post_scale_train_target_sum": post_scale_info["train_target_sum"],
            "global_post_scale_train_pred_sum": post_scale_info["train_pred_sum"],
            "selective_post_mode": selective_post_mode_this_well,
            "selective_post_target_wells": sorted(selective_post_target_wells),
            "selective_post_feature": args.selective_post_feature,
            "selective_post_feature_2": args.selective_post_feature_2,
            "selective_post_quantile": args.selective_post_quantile,
            "selective_post_quantile_2": args.selective_post_quantile_2,
            "selective_post_threshold": selective_post_strategy["threshold"],
            "selective_post_threshold_2": selective_post_strategy["threshold_2"],
            "selective_post_lowconf_scale": selective_post_strategy["lowconf_scale"],
            "selective_post_num_lowconf_segments": int(np.sum(lowconf_mask.astype(np.int64))),
            "soft_negative_weight": args.soft_negative_weight,
            "pred_min_points_per_segment": args.pred_min_points_per_segment,
            "rounding_mode": args.rounding_mode,
            "num_train_segments": int(len(train_df)),
            "num_val_segments": int(len(val_df)),
            "segment_count_mae": float(pred_segment_df["SegmentCountAbsError"].mean()),
            "segment_count_rmse": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            pred_segment_df["PredPointCount"].to_numpy(dtype=np.float64)
                            - pred_segment_df["RawGTCountInPredSegment"].to_numpy(dtype=np.float64)
                        )
                    )
                )
            ),
            "N_gt_points": int(len(raw_gt_depths)),
            "N_pred_points": int(len(pred_depths)),
            "count_diff": int(len(pred_depths) - len(raw_gt_depths)),
            "pred_to_gt_mean_dist": pred_to_gt["mean"],
            "pred_to_gt_median_dist": pred_to_gt["median"],
            "pred_to_gt_p90_dist": pred_to_gt["p90"],
            "gt_to_pred_mean_dist": gt_to_pred["mean"],
            "gt_to_pred_median_dist": gt_to_pred["median"],
            "gt_to_pred_p90_dist": gt_to_pred["p90"],
            "raw_points_covered_by_pred_segments": int(well_data["raw_points_covered_by_pred_segments"]),
            "raw_points_missed_outside_pred_segments": int(well_data["raw_points_missed_outside_pred_segments"]),
            "pred_segments_with_raw_gt_0": int((pred_segment_df["RawGTCountInPredSegment"] <= 0).sum()),
            "pred_segments_overlap_gt_dev_and_raw_gt_0": int(
                ((pred_segment_df["RawGTCountInPredSegment"] <= 0) & (pred_segment_df["GTDevOverlapFlag"] == 1)).sum()
            ),
        }
        with open(well_save_dir / "metrics.json", "w", encoding="utf-8") as file_obj:
            json.dump(metrics, file_obj, ensure_ascii=False, indent=2)
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise ValueError("No validation summary rows were produced")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(coef_rows).to_csv(save_dir / "loo_model_coefficients.csv", index=False, encoding="utf-8-sig")

    with open(save_dir / "config.json", "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "exp_id": args.exp_id,
                "exist_exp_dir": str(args.exist_exp_dir),
                "sample_dir": str(args.sample_dir),
                "raw_label_dir": str(args.raw_label_dir),
                "well_names": selected_wells,
                "gt_dev_rule": args.gt_dev_rule,
                "feature_cols": feature_cols,
                "feature_preset": args.feature_preset,
                "prob_weight_gamma": args.prob_weight_gamma,
                "fallback_count_mode": fallback_count_mode,
                "count_fusion_mode": args.count_fusion_mode,
                "fusion_learned_weight": args.fusion_learned_weight,
                "global_post_scale_mode": global_post_scale_mode,
                "global_post_scale_min": args.global_post_scale_min,
                "global_post_scale_max": args.global_post_scale_max,
                "selective_post_mode": selective_post_mode,
                "selective_post_target_wells": sorted(selective_post_target_wells),
                "selective_post_feature": args.selective_post_feature,
                "selective_post_feature_2": args.selective_post_feature_2,
                "selective_post_quantile": args.selective_post_quantile,
                "selective_post_quantile_2": args.selective_post_quantile_2,
                "soft_negative_weight": args.soft_negative_weight,
                "pred_min_points_per_segment": args.pred_min_points_per_segment,
                "rounding_mode": args.rounding_mode,
                "model_name": args.model_name,
                "model_alpha": args.model_alpha,
                "model_max_iter": args.model_max_iter,
                "tweedie_power": args.tweedie_power,
                "hgb_learning_rate": args.hgb_learning_rate,
                "hgb_max_depth": args.hgb_max_depth,
                "xgb_n_estimators": args.xgb_n_estimators,
                "xgb_max_depth": args.xgb_max_depth,
                "xgb_learning_rate": args.xgb_learning_rate,
                "xgb_subsample": args.xgb_subsample,
                "xgb_colsample_bytree": args.xgb_colsample_bytree,
                "save_dir": str(save_dir),
                "summary_csv": str(summary_csv),
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    with summary_csv.open("r", encoding="utf-8-sig", newline="") as file_obj:
        results = list(csv.DictReader(file_obj))
    append_result_to_docx(
        Path(args.docx_path),
        f"实验 {args.exp_id}",
        {
            "exist_exp_dir": str(args.exist_exp_dir),
            "sample_dir": str(args.sample_dir),
            "raw_label_dir": str(args.raw_label_dir),
            "gt_dev_rule": args.gt_dev_rule,
            "model_name": args.model_name,
            "model_alpha": args.model_alpha,
            "model_max_iter": args.model_max_iter,
            "tweedie_power": args.tweedie_power,
            "hgb_learning_rate": args.hgb_learning_rate,
            "hgb_max_depth": args.hgb_max_depth,
            "feature_preset": args.feature_preset,
            "feature_cols": feature_cols,
            "prob_weight_gamma": args.prob_weight_gamma,
            "fallback_count_mode": fallback_count_mode,
            "count_fusion_mode": args.count_fusion_mode,
            "fusion_learned_weight": args.fusion_learned_weight,
            "global_post_scale_mode": global_post_scale_mode,
            "global_post_scale_min": args.global_post_scale_min,
            "global_post_scale_max": args.global_post_scale_max,
            "selective_post_mode": selective_post_mode,
            "selective_post_target_wells": sorted(selective_post_target_wells),
            "selective_post_feature": args.selective_post_feature,
            "selective_post_feature_2": args.selective_post_feature_2,
            "selective_post_quantile": args.selective_post_quantile,
            "selective_post_quantile_2": args.selective_post_quantile_2,
            "soft_negative_weight": args.soft_negative_weight,
            "pred_min_points_per_segment": args.pred_min_points_per_segment,
            "rounding_mode": args.rounding_mode,
            "save_dir": str(save_dir),
            "summary_csv": str(summary_csv),
            "xgb_n_estimators": args.xgb_n_estimators,
            "xgb_max_depth": args.xgb_max_depth,
            "xgb_learning_rate": args.xgb_learning_rate,
            "xgb_subsample": args.xgb_subsample,
            "xgb_colsample_bytree": args.xgb_colsample_bytree,
        },
        results,
    )

    print(
        json.dumps(
            {
                "exp_id": args.exp_id,
                "save_dir": str(save_dir),
                "summary_csv": str(summary_csv),
                "feature_cols": feature_cols,
                "results": results,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_run_config_dict(
    args,
    selected_wells: list[str],
    feature_cols: list[str],
    fallback_count_mode: str,
    global_post_scale_mode: str,
    selective_post_mode: str,
    selective_post_target_wells: set[str],
    train_distance_csv: Path | None,
    save_dir: Path,
    summary_csv: Path,
    saved_model_root: Path | None = None,
) -> dict:
    config_dict = {
        "exp_id": args.exp_id,
        "run_mode": args.run_mode,
        "exist_exp_dir": str(args.exist_exp_dir),
        "sample_dir": str(args.sample_dir),
        "raw_label_dir": str(args.raw_label_dir),
        "well_names": selected_wells,
        "gt_dev_rule": args.gt_dev_rule,
        "feature_cols": feature_cols,
        "feature_preset": args.feature_preset,
        "prob_weight_gamma": args.prob_weight_gamma,
        "fallback_count_mode": fallback_count_mode,
        "count_fusion_mode": args.count_fusion_mode,
        "fusion_learned_weight": args.fusion_learned_weight,
        "global_post_scale_mode": global_post_scale_mode,
        "global_post_scale_min": args.global_post_scale_min,
        "global_post_scale_max": args.global_post_scale_max,
        "selective_post_mode": selective_post_mode,
        "selective_post_target_wells": sorted(selective_post_target_wells),
        "selective_post_feature": args.selective_post_feature,
        "selective_post_feature_2": args.selective_post_feature_2,
        "selective_post_quantile": args.selective_post_quantile,
        "selective_post_quantile_2": args.selective_post_quantile_2,
        "selective_post_max_pred_float": args.selective_post_max_pred_float,
        "train_distance_weight_mode": validate_train_distance_weight_mode(str(args.train_distance_weight_mode)),
        "train_distance_weight_power": args.train_distance_weight_power,
        "train_distance_weight_min": args.train_distance_weight_min,
        "train_distance_weight_max": args.train_distance_weight_max,
        "train_distance_csv": str(train_distance_csv) if train_distance_csv is not None else "",
        "boundary_expand_mode": validate_boundary_expand_mode(str(args.boundary_expand_mode)),
        "boundary_expand_prob_min": args.boundary_expand_prob_min,
        "boundary_expand_max_steps": args.boundary_expand_max_steps,
        "boundary_expand_max_depth": args.boundary_expand_max_depth,
        "soft_negative_weight": args.soft_negative_weight,
        "pred_min_points_per_segment": args.pred_min_points_per_segment,
        "rounding_mode": args.rounding_mode,
        "model_name": args.model_name,
        "model_alpha": args.model_alpha,
        "model_max_iter": args.model_max_iter,
        "tweedie_power": args.tweedie_power,
        "hgb_learning_rate": args.hgb_learning_rate,
        "hgb_max_depth": args.hgb_max_depth,
        "xgb_n_estimators": args.xgb_n_estimators,
        "xgb_max_depth": args.xgb_max_depth,
        "xgb_learning_rate": args.xgb_learning_rate,
        "xgb_subsample": args.xgb_subsample,
        "xgb_colsample_bytree": args.xgb_colsample_bytree,
        "save_model_artifacts": int(args.save_model_artifacts),
        "save_dir": str(save_dir),
        "summary_csv": str(summary_csv),
    }
    if saved_model_root is not None:
        config_dict["saved_model_root"] = str(saved_model_root)
    return config_dict


def fit_segment_refine_artifact(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    args,
    fallback_count_mode: str,
    global_post_scale_mode: str,
    selective_post_mode_this_well: str,
    artifact_type: str,
    artifact_target: str,
    train_wells: list[str],
    distance_weight_info: dict,
    train_distance_csv: Path | None,
) -> tuple[dict, list[dict]]:
    x_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["RawGTCountInPredSegment"].fillna(0.0).clip(lower=0.0)
    sample_weight = train_df["TrainSampleWeightUsed"].fillna(1.0).to_numpy(dtype=np.float64)

    learned_train_pred_count_float = np.zeros(len(train_df), dtype=np.float64)
    regressor = None
    coef_rows: list[dict] = []
    if str(args.model_name) != "rule_only":
        regressor = build_regressor(
            model_name=str(args.model_name),
            alpha=float(args.model_alpha),
            max_iter=int(args.model_max_iter),
            tweedie_power=float(args.tweedie_power),
            xgb_n_estimators=int(args.xgb_n_estimators),
            xgb_max_depth=int(args.xgb_max_depth),
            xgb_learning_rate=float(args.xgb_learning_rate),
            xgb_subsample=float(args.xgb_subsample),
            xgb_colsample_bytree=float(args.xgb_colsample_bytree),
            hgb_learning_rate=float(args.hgb_learning_rate),
            hgb_max_depth=int(args.hgb_max_depth),
        )
        regressor = fit_regressor(regressor, x_train, y_train, sample_weight)
        learned_train_pred_count_float = np.clip(regressor.predict(x_train), a_min=0.0, a_max=None)
        coef_rows.extend(build_coef_rows(artifact_target, regressor, feature_cols))

    fallback_basis_col = ""
    fallback_scale = np.nan
    fallback_train_pred_count_float = None
    if fallback_count_mode:
        fallback_basis_col, fallback_scale = fit_basis_scale(train_df, fallback_count_mode)
        fallback_train_pred_count_float = np.clip(
            train_df[fallback_basis_col].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64) * fallback_scale,
            a_min=0.0,
            a_max=None,
        )

    if str(args.model_name) == "rule_only":
        if fallback_train_pred_count_float is None:
            raise ValueError("model_name=rule_only requires fallback_count_mode")
        train_pred_count_float_before_post_scale = fallback_train_pred_count_float
    else:
        train_pred_count_float_before_post_scale = fuse_count_predictions(
            learned_pred=learned_train_pred_count_float,
            fallback_pred=fallback_train_pred_count_float,
            fusion_mode=str(args.count_fusion_mode),
            learned_weight=float(args.fusion_learned_weight),
        )

    post_scale_info = fit_global_post_scale(
        train_target=train_df["RawGTCountInPredSegment"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64),
        train_pred=train_pred_count_float_before_post_scale,
        mode=global_post_scale_mode,
        scale_min=float(args.global_post_scale_min),
        scale_max=float(args.global_post_scale_max),
    )
    train_pred_count_float_after_global_scale = apply_global_post_scale(
        train_pred_count_float_before_post_scale,
        post_scale_info,
    )
    selective_post_strategy = fit_selective_post_strategy(
        train_df=train_df,
        train_pred_after_global_scale=train_pred_count_float_after_global_scale,
        mode=selective_post_mode_this_well,
        feature_name=str(args.selective_post_feature),
        feature_name_2=str(args.selective_post_feature_2),
        quantile=float(args.selective_post_quantile),
        quantile_2=float(args.selective_post_quantile_2),
        fallback_scale=float(post_scale_info["raw_scale"]),
        max_pred_float=float(args.selective_post_max_pred_float),
    )

    artifact = {
        "artifact_version": 1,
        "artifact_type": artifact_type,
        "artifact_target": artifact_target,
        "train_wells": list(train_wells),
        "feature_cols": list(feature_cols),
        "model_name": str(args.model_name),
        "model_alpha": float(args.model_alpha),
        "model_max_iter": int(args.model_max_iter),
        "tweedie_power": float(args.tweedie_power),
        "hgb_learning_rate": float(args.hgb_learning_rate),
        "hgb_max_depth": int(args.hgb_max_depth),
        "xgb_n_estimators": int(args.xgb_n_estimators),
        "xgb_max_depth": int(args.xgb_max_depth),
        "xgb_learning_rate": float(args.xgb_learning_rate),
        "xgb_subsample": float(args.xgb_subsample),
        "xgb_colsample_bytree": float(args.xgb_colsample_bytree),
        "fallback_count_mode": fallback_count_mode,
        "fallback_basis_col": fallback_basis_col,
        "fallback_scale": float(fallback_scale) if np.isfinite(fallback_scale) else np.nan,
        "count_fusion_mode": str(args.count_fusion_mode),
        "fusion_learned_weight": float(args.fusion_learned_weight),
        "global_post_scale_info": post_scale_info,
        "selective_post_strategy": selective_post_strategy,
        "pred_min_points_per_segment": int(args.pred_min_points_per_segment),
        "rounding_mode": str(args.rounding_mode),
        "prob_weight_gamma": float(args.prob_weight_gamma),
        "boundary_expand_mode": validate_boundary_expand_mode(str(args.boundary_expand_mode)),
        "boundary_expand_prob_min": float(args.boundary_expand_prob_min),
        "boundary_expand_max_steps": int(args.boundary_expand_max_steps),
        "boundary_expand_max_depth": float(args.boundary_expand_max_depth),
        "gt_dev_rule": str(args.gt_dev_rule),
        "soft_negative_weight": float(args.soft_negative_weight),
        "train_distance_weight_mode": str(distance_weight_info.get("mode", "none")),
        "train_distance_weight_power": float(distance_weight_info.get("weight_power", getattr(args, "train_distance_weight_power", 1.0))),
        "train_distance_weight_min": float(distance_weight_info.get("weight_min", getattr(args, "train_distance_weight_min", 1.0))),
        "train_distance_weight_max": float(distance_weight_info.get("weight_max", getattr(args, "train_distance_weight_max", 1.0))),
        "train_distance_csv": str(train_distance_csv) if train_distance_csv is not None else "",
        "train_distance_weight_map": dict(distance_weight_info.get("well_weights", {})),
        "train_distance_map": dict(distance_weight_info.get("well_distances", {})),
        "exist_exp_dir": str(args.exist_exp_dir),
        "sample_dir": str(args.sample_dir),
        "raw_label_dir": str(args.raw_label_dir),
        "regressor": regressor,
    }
    return artifact, coef_rows


def predict_segments_with_artifact(
    segment_df: pd.DataFrame,
    artifact: dict,
    include_gt_columns: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    feature_cols = list(artifact["feature_cols"])
    validate_feature_cols(feature_cols, segment_df)
    x_val = segment_df[feature_cols].fillna(0.0)

    learned_pred_count_float = np.zeros(len(segment_df), dtype=np.float64)
    regressor = artifact.get("regressor")
    if str(artifact.get("model_name")) != "rule_only":
        if regressor is None:
            raise ValueError("Artifact does not contain a fitted regressor")
        learned_pred_count_float = np.clip(regressor.predict(x_val), a_min=0.0, a_max=None)

    fallback_basis_col = str(artifact.get("fallback_basis_col") or "")
    fallback_scale = artifact.get("fallback_scale", np.nan)
    fallback_pred_count_float = None
    if fallback_basis_col:
        if fallback_basis_col not in segment_df.columns:
            raise ValueError(f"Missing fallback basis col in inference dataset: {fallback_basis_col}")
        scale_value = float(fallback_scale) if np.isfinite(fallback_scale) else 1.0
        fallback_pred_count_float = np.clip(
            segment_df[fallback_basis_col].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64) * scale_value,
            a_min=0.0,
            a_max=None,
        )

    if str(artifact.get("model_name")) == "rule_only":
        if fallback_pred_count_float is None:
            raise ValueError("Saved rule_only artifact requires fallback basis data for inference")
        pred_count_float_before_post_scale = fallback_pred_count_float
    else:
        pred_count_float_before_post_scale = fuse_count_predictions(
            learned_pred=learned_pred_count_float,
            fallback_pred=fallback_pred_count_float,
            fusion_mode=str(artifact.get("count_fusion_mode", "learned_only")),
            learned_weight=float(artifact.get("fusion_learned_weight", 0.7)),
        )

    post_scale_info = artifact.get("global_post_scale_info", {"mode": "none", "scale": 1.0})
    pred_count_float_after_global_scale = apply_global_post_scale(
        pred_count_float_before_post_scale,
        post_scale_info,
    )
    selective_post_strategy = artifact.get(
        "selective_post_strategy",
        {
            "mode": "none",
            "feature_name": "",
            "feature_name_2": "",
            "quantile": np.nan,
            "quantile_2": np.nan,
            "threshold": np.nan,
            "threshold_2": np.nan,
            "lowconf_scale": 1.0,
        },
    )
    pred_count_float, lowconf_mask = apply_selective_post_strategy(
        pred_count_float_after_global_scale,
        segment_df,
        selective_post_strategy,
        pred_count_float_for_protection=pred_count_float_before_post_scale,
    )

    pred_segment_df = segment_df.copy()
    pred_segment_df["LearnedPredPointCountFloat"] = learned_pred_count_float
    pred_segment_df["FallbackCountBasisCol"] = fallback_basis_col
    pred_segment_df["FallbackCountScale"] = fallback_scale
    pred_segment_df["FallbackPredPointCountFloat"] = (
        fallback_pred_count_float if fallback_pred_count_float is not None else np.nan
    )
    pred_segment_df["PredPointCountFloatBeforePostScale"] = pred_count_float_before_post_scale
    pred_segment_df["GlobalPostScaleMode"] = post_scale_info.get("mode", "none")
    pred_segment_df["GlobalPostScaleRawScale"] = post_scale_info.get("raw_scale", np.nan)
    pred_segment_df["GlobalPostScaleAppliedScale"] = post_scale_info.get("scale", 1.0)
    pred_segment_df["GlobalPostScaleTrainTargetSum"] = post_scale_info.get("train_target_sum", np.nan)
    pred_segment_df["GlobalPostScaleTrainPredSum"] = post_scale_info.get("train_pred_sum", np.nan)
    pred_segment_df["SelectivePostMode"] = selective_post_strategy.get("mode", "none")
    pred_segment_df["SelectivePostFeature"] = selective_post_strategy.get("feature_name", "")
    pred_segment_df["SelectivePostFeature2"] = selective_post_strategy.get("feature_name_2", "")
    pred_segment_df["SelectivePostThreshold"] = selective_post_strategy.get("threshold", np.nan)
    pred_segment_df["SelectivePostThreshold2"] = selective_post_strategy.get("threshold_2", np.nan)
    pred_segment_df["SelectivePostQuantile"] = selective_post_strategy.get("quantile", np.nan)
    pred_segment_df["SelectivePostQuantile2"] = selective_post_strategy.get("quantile_2", np.nan)
    pred_segment_df["SelectivePostLowconfScale"] = selective_post_strategy.get("lowconf_scale", 1.0)
    pred_segment_df["SelectivePostMaxPredFloat"] = selective_post_strategy.get("max_pred_float", np.nan)
    pred_segment_df["SelectivePostIsLowConf"] = lowconf_mask.astype(int)
    pred_segment_df["PredPointCountFloatBeforeSelectivePost"] = pred_count_float_after_global_scale
    pred_segment_df["PredPointCountFloat"] = pred_count_float
    pred_segment_df["PredPointCount"] = pred_segment_df["PredPointCountFloat"].apply(
        lambda value: compute_pred_count(
            raw_value=float(value),
            pred_min_points_per_segment=int(artifact.get("pred_min_points_per_segment", 0)),
            rounding_mode=str(artifact.get("rounding_mode", "round")),
        )
    )
    if include_gt_columns and "RawGTCountInPredSegment" in pred_segment_df.columns:
        pred_segment_df["SegmentCountAbsError"] = (
            pred_segment_df["PredPointCount"] - pred_segment_df["RawGTCountInPredSegment"]
        ).abs()
    else:
        pred_segment_df["SegmentCountAbsError"] = np.nan
    return pred_segment_df, lowconf_mask


def save_model_artifact(model_dir: Path, artifact: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_dir / MODEL_ARTIFACT_FILENAME)

    meta = dict(artifact)
    regressor = meta.pop("regressor", None)
    meta["regressor_class"] = None if regressor is None else regressor.__class__.__name__
    with (model_dir / MODEL_META_FILENAME).open("w", encoding="utf-8") as file_obj:
        json.dump(to_jsonable(meta), file_obj, ensure_ascii=False, indent=2)


def load_model_artifact(model_dir: Path) -> dict:
    artifact_path = model_dir / MODEL_ARTIFACT_FILENAME
    if not artifact_path.exists():
        raise FileNotFoundError(f"Saved model artifact not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError(f"Unexpected artifact payload in {artifact_path}")
    return artifact


def resolve_saved_model_root(args, save_dir: Path) -> Path:
    return Path(args.saved_model_root) if args.saved_model_root else save_dir


def run_loo_eval(args) -> int:
    save_dir = Path(args.result_dir) if args.result_dir else (BASE_SAVE_DIR / sanitize(args.exp_id))
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = save_dir / "raw_point_guided_refine_summary.csv"

    verify_logs = dataset_builder.POINT.list_verify_logs(str(args.exist_exp_dir))
    selected_wells = parse_list_arg(args.well_names) if args.well_names else sorted(verify_logs.keys())
    missing_wells = [well for well in selected_wells if well not in verify_logs]
    if missing_wells:
        raise ValueError(f"Wells not found in verify logs: {missing_wells}")

    feature_cols = resolve_feature_cols(str(args.feature_preset), str(args.feature_cols))
    fallback_count_mode = validate_optional_count_mode(str(args.fallback_count_mode))
    global_post_scale_mode = validate_global_post_scale_mode(str(args.global_post_scale_mode))
    selective_post_mode = validate_selective_post_mode(str(args.selective_post_mode))
    selective_post_target_wells = set(parse_list_arg(args.selective_post_target_wells))
    boundary_expand_mode = validate_boundary_expand_mode(str(args.boundary_expand_mode))
    train_distance_weight_mode = validate_train_distance_weight_mode(str(args.train_distance_weight_mode))
    train_distance_csv, train_distance_df = resolve_train_distance_resources(args)
    config = dataset_builder.make_config(
        exist_exp_dir=str(args.exist_exp_dir),
        prob_weight_gamma=float(args.prob_weight_gamma),
        gt_min_points_per_segment=1,
        gt_rounding_mode="round",
        density_gt_col="P10",
    )

    per_well = {}
    all_rows = []
    dataset_save_dir = save_dir / "segment_dataset"
    dataset_save_dir.mkdir(parents=True, exist_ok=True)
    for well_name in selected_wells:
        well_data = build_well_segment_dataset(
            exist_csv=verify_logs[well_name],
            well_name=well_name,
            config=config,
            sample_dir=Path(args.sample_dir),
            raw_label_dir=Path(args.raw_label_dir),
            gt_dev_rule=str(args.gt_dev_rule),
            boundary_expand_mode=boundary_expand_mode,
            boundary_expand_prob_min=float(args.boundary_expand_prob_min),
            boundary_expand_max_steps=int(args.boundary_expand_max_steps),
            boundary_expand_max_depth=float(args.boundary_expand_max_depth),
        )
        well_data["segment_df"] = compute_target_weights(
            well_data["segment_df"],
            soft_negative_weight=float(args.soft_negative_weight),
        )
        per_well[well_name] = well_data
        well_dir = dataset_save_dir / well_name
        well_dir.mkdir(parents=True, exist_ok=True)
        well_data["segment_df"].to_csv(well_dir / "segment_dataset.csv", index=False, encoding="utf-8-sig")
        well_data["raw_gt_points"].to_csv(well_dir / "raw_gt_points.csv", index=False, encoding="utf-8-sig")
        well_data["gt_dev_segments"].to_csv(well_dir / "gt_dev_segments.csv", index=False, encoding="utf-8-sig")
        if not well_data["segment_df"].empty:
            all_rows.append(well_data["segment_df"])

    dataset_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if dataset_df.empty:
        raise ValueError("Segment dataset is empty")
    dataset_df.to_csv(dataset_save_dir / "all_segment_dataset.csv", index=False, encoding="utf-8-sig")
    validate_feature_cols(feature_cols, dataset_df)

    summary_rows = []
    coef_rows = []
    saved_model_root = resolve_saved_model_root(args, save_dir)
    if int(args.save_model_artifacts):
        saved_model_root.mkdir(parents=True, exist_ok=True)

    for well_name in selected_wells:
        val_df = per_well[well_name]["segment_df"].copy()
        if val_df.empty:
            continue
        train_parts = [per_well[other]["segment_df"] for other in selected_wells if other != well_name]
        train_parts = [part for part in train_parts if not part.empty]
        if not train_parts:
            raise ValueError(f"No training segments available for {well_name}")
        train_df = pd.concat(train_parts, ignore_index=True)
        train_wells = [other for other in selected_wells if other != well_name]
        distance_weight_info = build_train_distance_weight_info(
            target_well=well_name,
            train_wells=train_wells,
            train_distance_weight_mode=train_distance_weight_mode,
            distance_df=train_distance_df,
            weight_power=float(args.train_distance_weight_power),
            weight_min=float(args.train_distance_weight_min),
            weight_max=float(args.train_distance_weight_max),
        )
        train_df = apply_train_distance_weight_info(train_df, distance_weight_info)

        selective_post_mode_this_well = (
            selective_post_mode
            if (not selective_post_target_wells or well_name in selective_post_target_wells)
            else "none"
        )
        artifact, artifact_coef_rows = fit_segment_refine_artifact(
            train_df=train_df,
            feature_cols=feature_cols,
            args=args,
            fallback_count_mode=fallback_count_mode,
            global_post_scale_mode=global_post_scale_mode,
            selective_post_mode_this_well=selective_post_mode_this_well,
            artifact_type="loo_split_model",
            artifact_target=well_name,
            train_wells=train_wells,
            distance_weight_info=distance_weight_info,
            train_distance_csv=train_distance_csv,
        )
        coef_rows.extend(artifact_coef_rows)

        pred_segment_df, lowconf_mask = predict_segments_with_artifact(
            segment_df=val_df,
            artifact=artifact,
            include_gt_columns=True,
        )

        well_data = per_well[well_name]
        pred_points = build_pred_points_from_segment_df(
            df_sorted=well_data["df_sorted"],
            payloads=well_data["payloads"],
            pred_segment_df=pred_segment_df,
            config=config,
        )
        raw_gt_points = well_data["raw_gt_points"]
        raw_gt_depths = well_data["raw_gt_depths"]
        pred_depths = (
            pred_points[config.depth_col].to_numpy(dtype=np.float64)
            if not pred_points.empty
            else np.array([], dtype=np.float64)
        )
        pred_to_gt = dataset_builder.POINT.nearest_distance_stats(pred_depths, raw_gt_depths)
        gt_to_pred = dataset_builder.POINT.nearest_distance_stats(raw_gt_depths, pred_depths)

        well_save_dir = save_dir / well_name
        well_save_dir.mkdir(parents=True, exist_ok=True)
        raw_gt_points.to_csv(well_save_dir / "raw_gt_points.csv", index=False, encoding="utf-8-sig")
        well_data["gt_dev_segments"].to_csv(well_save_dir / "gt_dev_segments.csv", index=False, encoding="utf-8-sig")
        pred_points.to_csv(well_save_dir / "pred_fracture_points.csv", index=False, encoding="utf-8-sig")
        pred_segment_df.to_csv(well_save_dir / "pred_segment_summary.csv", index=False, encoding="utf-8-sig")
        train_df.to_csv(well_save_dir / "train_segment_dataset.csv", index=False, encoding="utf-8-sig")

        if int(args.save_model_artifacts):
            save_model_artifact(saved_model_root / format_verify_well_dir_name(well_name), artifact)

        metrics = {
            "well": well_name,
            "exist_csv": well_data["exist_csv"],
            "model_name": args.model_name,
            "model_alpha": args.model_alpha,
            "model_max_iter": args.model_max_iter,
            "tweedie_power": args.tweedie_power,
            "hgb_learning_rate": args.hgb_learning_rate,
            "hgb_max_depth": args.hgb_max_depth,
            "feature_preset": args.feature_preset,
            "feature_cols": feature_cols,
            "prob_weight_gamma": args.prob_weight_gamma,
            "fallback_count_mode": fallback_count_mode,
            "count_fusion_mode": args.count_fusion_mode,
            "fusion_learned_weight": args.fusion_learned_weight,
            "global_post_scale_mode": global_post_scale_mode,
            "global_post_scale_raw_scale": pred_segment_df["GlobalPostScaleRawScale"].iloc[0] if not pred_segment_df.empty else np.nan,
            "global_post_scale_applied_scale": pred_segment_df["GlobalPostScaleAppliedScale"].iloc[0] if not pred_segment_df.empty else np.nan,
            "global_post_scale_train_target_sum": pred_segment_df["GlobalPostScaleTrainTargetSum"].iloc[0] if not pred_segment_df.empty else np.nan,
            "global_post_scale_train_pred_sum": pred_segment_df["GlobalPostScaleTrainPredSum"].iloc[0] if not pred_segment_df.empty else np.nan,
            "selective_post_mode": selective_post_mode_this_well,
            "selective_post_target_wells": sorted(selective_post_target_wells),
            "selective_post_feature": args.selective_post_feature,
            "selective_post_feature_2": args.selective_post_feature_2,
            "selective_post_quantile": args.selective_post_quantile,
            "selective_post_quantile_2": args.selective_post_quantile_2,
            "selective_post_threshold": pred_segment_df["SelectivePostThreshold"].iloc[0] if not pred_segment_df.empty else np.nan,
            "selective_post_threshold_2": pred_segment_df["SelectivePostThreshold2"].iloc[0] if not pred_segment_df.empty else np.nan,
            "selective_post_lowconf_scale": pred_segment_df["SelectivePostLowconfScale"].iloc[0] if not pred_segment_df.empty else np.nan,
            "selective_post_max_pred_float": pred_segment_df["SelectivePostMaxPredFloat"].iloc[0] if not pred_segment_df.empty else np.nan,
            "selective_post_num_lowconf_segments": int(np.sum(lowconf_mask.astype(np.int64))),
            "train_distance_weight_mode": train_distance_weight_mode,
            "train_distance_weight_power": float(args.train_distance_weight_power),
            "train_distance_weight_min": float(args.train_distance_weight_min),
            "train_distance_weight_max": float(args.train_distance_weight_max),
            "train_distance_csv": str(train_distance_csv) if train_distance_csv is not None else "",
            "train_distance_mean_weight": float(train_df["TrainWellDistanceWeight"].mean()),
            "train_distance_min_weight": float(train_df["TrainWellDistanceWeight"].min()),
            "train_distance_max_weight": float(train_df["TrainWellDistanceWeight"].max()),
            "train_distance_weight_map": distance_weight_info.get("well_weights", {}),
            "train_distance_map": distance_weight_info.get("well_distances", {}),
            "boundary_expand_mode": boundary_expand_mode,
            "boundary_expand_prob_min": float(args.boundary_expand_prob_min),
            "boundary_expand_max_steps": int(args.boundary_expand_max_steps),
            "boundary_expand_max_depth": float(args.boundary_expand_max_depth),
            "boundary_base_positive_samples": int(well_data["boundary_expand_stats"]["boundary_base_positive_samples"]),
            "boundary_effective_positive_samples": int(well_data["boundary_expand_stats"]["boundary_effective_positive_samples"]),
            "boundary_added_positive_samples": int(well_data["boundary_expand_stats"]["boundary_added_positive_samples"]),
            "boundary_base_segment_count": int(well_data["boundary_expand_stats"]["boundary_base_segment_count"]),
            "boundary_effective_segment_count": int(well_data["boundary_expand_stats"]["boundary_effective_segment_count"]),
            "boundary_added_left_samples": int(well_data["boundary_expand_stats"]["boundary_added_left_samples"]),
            "boundary_added_right_samples": int(well_data["boundary_expand_stats"]["boundary_added_right_samples"]),
            "soft_negative_weight": args.soft_negative_weight,
            "pred_min_points_per_segment": args.pred_min_points_per_segment,
            "rounding_mode": args.rounding_mode,
            "num_train_segments": int(len(train_df)),
            "num_val_segments": int(len(val_df)),
            "segment_count_mae": float(pred_segment_df["SegmentCountAbsError"].mean()),
            "segment_count_rmse": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            pred_segment_df["PredPointCount"].to_numpy(dtype=np.float64)
                            - pred_segment_df["RawGTCountInPredSegment"].to_numpy(dtype=np.float64)
                        )
                    )
                )
            ),
            "N_gt_points": int(len(raw_gt_depths)),
            "N_pred_points": int(len(pred_depths)),
            "count_diff": int(len(pred_depths) - len(raw_gt_depths)),
            "count_error_pct": compute_count_error_pct(int(len(pred_depths) - len(raw_gt_depths)), int(len(raw_gt_depths))),
            "pred_to_gt_mean_dist": pred_to_gt["mean"],
            "pred_to_gt_median_dist": pred_to_gt["median"],
            "pred_to_gt_p90_dist": pred_to_gt["p90"],
            "gt_to_pred_mean_dist": gt_to_pred["mean"],
            "gt_to_pred_median_dist": gt_to_pred["median"],
            "gt_to_pred_p90_dist": gt_to_pred["p90"],
            "raw_points_covered_by_pred_segments": int(well_data["raw_points_covered_by_pred_segments"]),
            "raw_points_missed_outside_pred_segments": int(well_data["raw_points_missed_outside_pred_segments"]),
            "pred_segments_with_raw_gt_0": int((pred_segment_df["RawGTCountInPredSegment"] <= 0).sum()),
            "pred_segments_overlap_gt_dev_and_raw_gt_0": int(
                ((pred_segment_df["RawGTCountInPredSegment"] <= 0) & (pred_segment_df["GTDevOverlapFlag"] == 1)).sum()
            ),
            "saved_model_dir": str(
                (saved_model_root / format_verify_well_dir_name(well_name)) if int(args.save_model_artifacts) else ""
            ),
        }
        with open(well_save_dir / "metrics.json", "w", encoding="utf-8") as file_obj:
            json.dump(metrics, file_obj, ensure_ascii=False, indent=2)
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise ValueError("No validation summary rows were produced")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(coef_rows).to_csv(save_dir / "loo_model_coefficients.csv", index=False, encoding="utf-8-sig")
    overall_metrics = {
        "num_wells": int(len(summary_df)),
        "total_gt_points": int(summary_df["N_gt_points"].fillna(0).sum()),
        "total_pred_points": int(summary_df["N_pred_points"].fillna(0).sum()),
        "total_abs_count_error": int(summary_df["count_diff"].abs().fillna(0).sum()),
        "overall_count_error_pct": compute_count_error_pct(
            float(summary_df["count_diff"].abs().fillna(0).sum()),
            float(summary_df["N_gt_points"].fillna(0).sum()),
        ),
        "avg_count_error_pct": float(summary_df["count_error_pct"].dropna().mean()) if "count_error_pct" in summary_df.columns else np.nan,
    }
    with open(save_dir / "overall_metrics.json", "w", encoding="utf-8") as file_obj:
        json.dump(to_jsonable(overall_metrics), file_obj, ensure_ascii=False, indent=2)

    run_config = build_run_config_dict(
        args=args,
        selected_wells=selected_wells,
        feature_cols=feature_cols,
        fallback_count_mode=fallback_count_mode,
        global_post_scale_mode=global_post_scale_mode,
        selective_post_mode=selective_post_mode,
        selective_post_target_wells=selective_post_target_wells,
        train_distance_csv=train_distance_csv,
        save_dir=save_dir,
        summary_csv=summary_csv,
        saved_model_root=saved_model_root if int(args.save_model_artifacts) else None,
    )
    with open(save_dir / "config.json", "w", encoding="utf-8") as file_obj:
        json.dump(to_jsonable(run_config), file_obj, ensure_ascii=False, indent=2)

    with summary_csv.open("r", encoding="utf-8-sig", newline="") as file_obj:
        results = list(csv.DictReader(file_obj))
    append_result_to_docx(
        Path(args.docx_path),
        f"实验 {args.exp_id}",
        run_config,
        results,
    )

    print(
        json.dumps(
            {
                "exp_id": args.exp_id,
                "run_mode": args.run_mode,
                "save_dir": str(save_dir),
                "summary_csv": str(summary_csv),
                "saved_model_root": str(saved_model_root) if int(args.save_model_artifacts) else "",
                "overall_metrics": overall_metrics,
                "feature_cols": feature_cols,
                "results": results,
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_predict(args) -> int:
    save_dir = Path(args.result_dir) if args.result_dir else (BASE_SAVE_DIR / sanitize(args.exp_id))
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_model_root = resolve_saved_model_root(args, save_dir)

    if not args.predict_exist_csv:
        raise ValueError("run_mode=predict requires --predict-exist-csv")
    if not args.predict_well_name:
        raise ValueError("run_mode=predict requires --predict-well-name")

    if args.load_model_dir:
        model_dir = Path(args.load_model_dir)
    else:
        if not args.load_model_well:
            raise ValueError("run_mode=predict requires --load-model-dir or --load-model-well")
        model_dir = saved_model_root / format_verify_well_dir_name(args.load_model_well)

    artifact = load_model_artifact(model_dir)
    config = dataset_builder.make_config(
        exist_exp_dir=str(artifact.get("exist_exp_dir") or args.exist_exp_dir),
        prob_weight_gamma=float(artifact.get("prob_weight_gamma", args.prob_weight_gamma)),
        gt_min_points_per_segment=1,
        gt_rounding_mode="round",
        density_gt_col="P10",
    )
    boundary_expand_mode = validate_boundary_expand_mode(
        str(artifact.get("boundary_expand_mode", args.boundary_expand_mode))
    )
    boundary_expand_prob_min = float(artifact.get("boundary_expand_prob_min", args.boundary_expand_prob_min))
    boundary_expand_max_steps = int(artifact.get("boundary_expand_max_steps", args.boundary_expand_max_steps))
    boundary_expand_max_depth = float(artifact.get("boundary_expand_max_depth", args.boundary_expand_max_depth))
    well_data = build_predict_segment_dataset(
        exist_csv=str(args.predict_exist_csv),
        well_name=str(args.predict_well_name),
        config=config,
        boundary_expand_mode=boundary_expand_mode,
        boundary_expand_prob_min=boundary_expand_prob_min,
        boundary_expand_max_steps=boundary_expand_max_steps,
        boundary_expand_max_depth=boundary_expand_max_depth,
    )
    if well_data["segment_df"].empty:
        raise ValueError(f"No predicted fracture segments found in: {args.predict_exist_csv}")

    pred_segment_df, lowconf_mask = predict_segments_with_artifact(
        segment_df=well_data["segment_df"].copy(),
        artifact=artifact,
        include_gt_columns=False,
    )
    pred_points = build_pred_points_from_segment_df(
        df_sorted=well_data["df_sorted"],
        payloads=well_data["payloads"],
        pred_segment_df=pred_segment_df,
        config=config,
    )

    inference_dir = save_dir / f"{args.predict_well_name}__with__{artifact.get('artifact_target', 'selected_model')}"
    inference_dir.mkdir(parents=True, exist_ok=True)
    pred_segment_df.to_csv(inference_dir / "pred_segment_summary.csv", index=False, encoding="utf-8-sig")
    pred_points.to_csv(inference_dir / "pred_fracture_points.csv", index=False, encoding="utf-8-sig")
    well_data["df"].to_csv(inference_dir / "input_predict_log.csv", index=False, encoding="utf-8-sig")

    inference_config = {
        "exp_id": args.exp_id,
        "run_mode": args.run_mode,
        "predict_exist_csv": str(args.predict_exist_csv),
        "predict_well_name": str(args.predict_well_name),
        "load_model_dir": str(model_dir),
        "artifact_target": artifact.get("artifact_target"),
        "artifact_type": artifact.get("artifact_type"),
        "train_wells": artifact.get("train_wells", []),
        "train_distance_weight_mode": artifact.get("train_distance_weight_mode", "none"),
        "train_distance_weight_power": artifact.get("train_distance_weight_power", 1.0),
        "train_distance_weight_min": artifact.get("train_distance_weight_min", 1.0),
        "train_distance_weight_max": artifact.get("train_distance_weight_max", 1.0),
        "train_distance_csv": artifact.get("train_distance_csv", ""),
        "train_distance_weight_map": artifact.get("train_distance_weight_map", {}),
        "train_distance_map": artifact.get("train_distance_map", {}),
        "feature_cols": artifact.get("feature_cols", []),
        "fallback_count_mode": artifact.get("fallback_count_mode", ""),
        "count_fusion_mode": artifact.get("count_fusion_mode", "learned_only"),
        "boundary_expand_mode": boundary_expand_mode,
        "boundary_expand_prob_min": boundary_expand_prob_min,
        "boundary_expand_max_steps": boundary_expand_max_steps,
        "boundary_expand_max_depth": boundary_expand_max_depth,
        "boundary_base_positive_samples": int(well_data["boundary_expand_stats"]["boundary_base_positive_samples"]),
        "boundary_effective_positive_samples": int(well_data["boundary_expand_stats"]["boundary_effective_positive_samples"]),
        "boundary_added_positive_samples": int(well_data["boundary_expand_stats"]["boundary_added_positive_samples"]),
        "boundary_base_segment_count": int(well_data["boundary_expand_stats"]["boundary_base_segment_count"]),
        "boundary_effective_segment_count": int(well_data["boundary_expand_stats"]["boundary_effective_segment_count"]),
        "selective_post_max_pred_float": artifact.get("selective_post_strategy", {}).get("max_pred_float", np.nan),
        "pred_min_points_per_segment": artifact.get("pred_min_points_per_segment", 0),
        "rounding_mode": artifact.get("rounding_mode", "round"),
        "num_segments": int(len(pred_segment_df)),
        "num_lowconf_segments": int(np.sum(lowconf_mask.astype(np.int64))),
        "num_pred_points": int(len(pred_points)),
        "save_dir": str(inference_dir),
    }
    with (inference_dir / "inference_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(to_jsonable(inference_config), file_obj, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "exp_id": args.exp_id,
                "run_mode": args.run_mode,
                "model_dir": str(model_dir),
                "predict_well_name": str(args.predict_well_name),
                "save_dir": str(inference_dir),
                "num_segments": int(len(pred_segment_df)),
                "num_pred_points": int(len(pred_points)),
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--run-mode", default="loo_eval", choices=["loo_eval", "predict"])
    parser.add_argument("--exist-exp-dir", default=str(DEFAULT_EXIST_EXP_DIR))
    parser.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--raw-label-dir", default=str(DEFAULT_RAW_LABEL_DIR))
    parser.add_argument("--docx-path", default=str(DEFAULT_DOCX_PATH))
    parser.add_argument("--result-dir")
    parser.add_argument("--well-names", default="")
    parser.add_argument("--gt-dev-rule", default="any_density")
    parser.add_argument("--prob-weight-gamma", type=float, default=2.0)
    parser.add_argument("--pred-min-points-per-segment", type=int, default=0)
    parser.add_argument("--rounding-mode", default="round")
    parser.add_argument("--fallback-count-mode", default="")
    parser.add_argument("--count-fusion-mode", default="learned_only")
    parser.add_argument("--fusion-learned-weight", type=float, default=0.7)
    parser.add_argument("--global-post-scale-mode", default="none")
    parser.add_argument("--global-post-scale-min", type=float, default=0.5)
    parser.add_argument("--global-post-scale-max", type=float, default=1.5)
    parser.add_argument("--selective-post-mode", default="none")
    parser.add_argument("--selective-post-target-wells", default="")
    parser.add_argument("--selective-post-feature", default="ProbMassPerLength")
    parser.add_argument("--selective-post-feature-2", default="ProbMean")
    parser.add_argument("--selective-post-quantile", type=float, default=0.5)
    parser.add_argument("--selective-post-quantile-2", type=float, default=0.5)
    parser.add_argument("--selective-post-max-pred-float", type=float, default=-1.0)
    parser.add_argument("--soft-negative-weight", type=float, default=0.5)
    parser.add_argument("--train-distance-weight-mode", default="none")
    parser.add_argument("--train-distance-csv", default="")
    parser.add_argument("--train-distance-weight-power", type=float, default=1.0)
    parser.add_argument("--train-distance-weight-min", type=float, default=0.7)
    parser.add_argument("--train-distance-weight-max", type=float, default=1.3)
    parser.add_argument("--boundary-expand-mode", default="none")
    parser.add_argument("--boundary-expand-prob-min", type=float, default=0.35)
    parser.add_argument("--boundary-expand-max-steps", type=int, default=2)
    parser.add_argument("--boundary-expand-max-depth", type=float, default=0.0)
    parser.add_argument("--model-name", default="tweedie")
    parser.add_argument("--model-alpha", type=float, default=0.03)
    parser.add_argument("--model-max-iter", type=int, default=3000)
    parser.add_argument("--tweedie-power", type=float, default=1.5)
    parser.add_argument("--feature-preset", default="high_density_compact_v1")
    parser.add_argument("--feature-cols", default=",".join(dataset_builder.DEFAULT_FEATURE_COLS))
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--hgb-max-depth", type=int, default=3)
    parser.add_argument("--save-model-artifacts", type=int, default=1)
    parser.add_argument("--saved-model-root", default="")
    parser.add_argument("--load-model-dir", default="")
    parser.add_argument("--load-model-well", default="")
    parser.add_argument("--predict-exist-csv", default="")
    parser.add_argument("--predict-well-name", default="")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.run_mode == "predict":
        return run_predict(args)
    return run_loo_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
