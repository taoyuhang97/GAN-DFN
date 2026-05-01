import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
POINT_REFINE_PATH = ROOT / "fracture_point_refine_by_density.py"
DEFAULT_EXIST_EXP_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝存在性预测/LSTM/单井验证/成像测井裂缝预测/cnn+lstm/exp40_missing_drop_ac_gr_3x3_seq5_AC_GR"
)
DEFAULT_SAVE_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝位置预测/基于密度的裂缝点位分析/裂缝点位精细化/segment_count_dataset"
)
HIGH_PROB_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
EXCESS_PROB_THRESHOLDS = (0.6, 0.8)
DEFAULT_FEATURE_COLS = [
    "LogSegLength",
    "SegLength",
    "ProbMean",
    "ProbMax",
    "ProbStd",
    "LogProbMass",
    "ProbMass",
    "HighProbLen_05",
    "HighProbLen_07",
    "HighProbLen_09",
    "HighProbFrac_07",
    "HighProbFrac_09",
    "Peakiness",
    "LogLengthXProbMass",
]
FEATURE_PRESETS = {
    "default": list(DEFAULT_FEATURE_COLS),
    "high_density_v1": [
        "LogSegLength",
        "SampleCount",
        "ProbMean",
        "ProbMax",
        "ProbStd",
        "ProbGammaMean",
        "ProbMass",
        "ProbMassRaw",
        "ProbMassPerLength",
        "HighProbLen_05",
        "HighProbLen_06",
        "HighProbLen_08",
        "HighProbFrac_05",
        "HighProbFrac_06",
        "HighProbFrac_08",
        "HighProbMass_06",
        "HighProbMass_08",
        "ProbExcessMass_06",
        "ProbExcessMass_08",
        "LengthXProbMean",
    ],
    "high_density_compact_v1": [
        "LogSegLength",
        "SampleCount",
        "ProbMean",
        "ProbMax",
        "ProbStd",
        "ProbMassPerLength",
        "HighProbLen_05",
        "HighProbLen_06",
        "HighProbLen_08",
        "HighProbFrac_05",
        "ProbExcessMass_06",
        "ProbExcessMass_08",
        "LengthXProbMean",
    ],
}


def load_point_refine_module():
    spec = importlib.util.spec_from_file_location(
        "fracture_point_refine_dynamic_module",
        POINT_REFINE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from: {POINT_REFINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POINT = load_point_refine_module()


def segment_integral(depth: np.ndarray, values: np.ndarray) -> float:
    if depth.size <= 1 or values.size <= 1:
        return 0.0
    return float(POINT.integrate_trapezoid(depth, values)[-1])


def safe_float(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(value)


def safe_positive(value: float) -> float:
    return max(safe_float(value), 0.0)


def safe_ratio(numerator: float, denominator: float) -> float:
    numerator = safe_positive(numerator)
    denominator = safe_positive(denominator)
    if denominator <= 1e-8:
        return 0.0
    return numerator / denominator


def high_prob_length(depth: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    mask = (prob >= threshold).astype(np.float64)
    return segment_integral(depth, mask)


def high_prob_mass(depth: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    values = np.where(prob >= threshold, prob, 0.0)
    return segment_integral(depth, values)


def prob_excess_mass(depth: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    values = np.clip(prob - threshold, a_min=0.0, a_max=None)
    return segment_integral(depth, values)


def make_config(
    exist_exp_dir: str,
    prob_weight_gamma: float,
    gt_min_points_per_segment: int,
    gt_rounding_mode: str,
    density_gt_col: str,
):
    return POINT.PointRefineConfig(
        exist_exp_dir=exist_exp_dir,
        density_exp_dir="",
        save_dir="",
        well_names=[],
        prob_weight_gamma=prob_weight_gamma,
        min_points_per_segment=gt_min_points_per_segment,
        rounding_mode=gt_rounding_mode,
        density_gt_col=density_gt_col,
        use_pred_density=False,
        segment_count_mode="density_integral",
        shape_mode="prob_only",
    )


def build_segment_row(
    well_name: str,
    payload: dict,
    df_sorted: pd.DataFrame,
    gt_depths: np.ndarray,
    config,
) -> dict:
    seg_df = df_sorted.iloc[payload["StartIdx"] : payload["EndIdx"] + 1].copy()
    depth_seg = payload["depth_seg"]
    prob_seg = (
        POINT.safe_to_numeric(seg_df[config.pred_prob_col])
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .to_numpy(dtype=np.float64)
    )
    prob_mean = safe_positive(float(np.mean(prob_seg))) if prob_seg.size > 0 else 0.0
    prob_max = safe_positive(float(np.max(prob_seg))) if prob_seg.size > 0 else 0.0
    prob_std = safe_positive(float(np.std(prob_seg))) if prob_seg.size > 0 else 0.0
    prob_mass_raw = safe_positive(segment_integral(depth_seg, prob_seg))
    prob_mass_gamma = safe_positive(payload["ProbMass"])
    seg_length = safe_positive(payload["SegLength"])
    high_prob_len = {
        threshold: safe_positive(high_prob_length(depth_seg, prob_seg, threshold))
        for threshold in HIGH_PROB_THRESHOLDS
    }
    high_prob_mass_map = {
        threshold: safe_positive(high_prob_mass(depth_seg, prob_seg, threshold))
        for threshold in EXCESS_PROB_THRESHOLDS
    }
    prob_excess_mass_map = {
        threshold: safe_positive(prob_excess_mass(depth_seg, prob_seg, threshold))
        for threshold in EXCESS_PROB_THRESHOLDS
    }
    gt_count = int(
        POINT.count_points_in_depth_range(
            gt_depths,
            payload["SegStartDepth"],
            payload["SegEndDepth"],
        )
    )

    row = {
        "WellName": well_name,
        "Segment_ID": int(payload["Segment_ID"]),
        "SegStartDepth": float(payload["SegStartDepth"]),
        "SegEndDepth": float(payload["SegEndDepth"]),
        "SegLength": seg_length,
        "SampleCount": int(len(depth_seg)),
        "ProbMean": prob_mean,
        "ProbMax": prob_max,
        "ProbStd": prob_std,
        "ProbRange": safe_positive(prob_max - prob_mean),
        "ProbGammaMean": safe_positive(payload["ProbGammaMean"]),
        "ProbMass": prob_mass_gamma,
        "ProbMassRaw": prob_mass_raw,
        "ProbMassPerLength": safe_ratio(prob_mass_gamma, seg_length),
        "ProbMassRawPerLength": safe_ratio(prob_mass_raw, seg_length),
        "LengthXProbMean": safe_positive(seg_length * prob_mean),
        "LengthXProbMass": safe_positive(seg_length * prob_mass_gamma),
        "HighProbLen_05": high_prob_len[0.5],
        "HighProbLen_06": high_prob_len[0.6],
        "HighProbLen_07": high_prob_len[0.7],
        "HighProbLen_08": high_prob_len[0.8],
        "HighProbLen_09": high_prob_len[0.9],
        "HighProbFrac_05": safe_ratio(high_prob_len[0.5], seg_length),
        "HighProbFrac_06": safe_ratio(high_prob_len[0.6], seg_length),
        "HighProbFrac_07": safe_ratio(high_prob_len[0.7], seg_length),
        "HighProbFrac_08": safe_ratio(high_prob_len[0.8], seg_length),
        "HighProbFrac_09": safe_ratio(high_prob_len[0.9], seg_length),
        "HighProbMass_06": high_prob_mass_map[0.6],
        "HighProbMass_08": high_prob_mass_map[0.8],
        "HighProbMassFrac_06": safe_ratio(high_prob_mass_map[0.6], prob_mass_raw),
        "HighProbMassFrac_08": safe_ratio(high_prob_mass_map[0.8], prob_mass_raw),
        "ProbExcessMass_06": prob_excess_mass_map[0.6],
        "ProbExcessMass_08": prob_excess_mass_map[0.8],
        "Peakiness": prob_max / prob_mean if prob_mean > 1e-8 else 0.0,
        "LogSegLength": float(np.log1p(seg_length)),
        "LogProbMass": float(np.log1p(prob_mass_gamma)),
        "LogLengthXProbMass": float(np.log1p(seg_length * prob_mass_gamma)),
        "LogProbExcessMass_06": float(np.log1p(prob_excess_mass_map[0.6])),
        "LogProbExcessMass_08": float(np.log1p(prob_excess_mass_map[0.8])),
        "GTPointCountInPredSegment": gt_count,
    }
    return row


def load_well_segments(exist_csv: str, well_name: str, config) -> dict:
    df = pd.read_csv(exist_csv, encoding="utf-8-sig")
    gt_points = POINT.build_fracture_points(
        df,
        mask_col=config.gt_label_col,
        density_col=config.density_gt_col,
        config=config,
        prob_col=None,
        use_prob_weighting=False,
    )
    gt_depths = (
        gt_points[config.depth_col].to_numpy(dtype=np.float64)
        if not gt_points.empty
        else np.array([], dtype=np.float64)
    )
    df_sorted, payloads = POINT.extract_segment_payloads(
        df,
        mask_col=config.pred_label_col,
        density_col=None,
        config=config,
        prob_col=config.pred_prob_col,
        shape_mode="prob_only",
    )
    rows = [
        build_segment_row(
            well_name=well_name,
            payload=payload,
            df_sorted=df_sorted,
            gt_depths=gt_depths,
            config=config,
        )
        for payload in payloads
    ]
    return {
        "well_name": well_name,
        "exist_csv": exist_csv,
        "df": df,
        "df_sorted": df_sorted,
        "payloads": payloads,
        "gt_points": gt_points,
        "gt_depths": gt_depths,
        "segment_df": pd.DataFrame(rows),
    }


def build_segment_dataset(
    exist_exp_dir: str,
    well_names: list[str] | None = None,
    save_dir: str | None = None,
    prob_weight_gamma: float = 2.0,
    gt_min_points_per_segment: int = 1,
    gt_rounding_mode: str = "round",
    density_gt_col: str = "P10",
):
    verify_logs = POINT.list_verify_logs(exist_exp_dir)
    selected_wells = sorted(verify_logs.keys()) if not well_names else [well for well in well_names if well in verify_logs]
    config = make_config(
        exist_exp_dir=exist_exp_dir,
        prob_weight_gamma=prob_weight_gamma,
        gt_min_points_per_segment=gt_min_points_per_segment,
        gt_rounding_mode=gt_rounding_mode,
        density_gt_col=density_gt_col,
    )

    all_rows = []
    per_well: dict[str, dict] = {}
    for well_name in selected_wells:
        well_data = load_well_segments(verify_logs[well_name], well_name, config)
        per_well[well_name] = well_data
        if not well_data["segment_df"].empty:
            all_rows.append(well_data["segment_df"])

    dataset_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        dataset_df.to_csv(save_path / "segment_count_dataset.csv", index=False, encoding="utf-8-sig")
        for well_name, well_data in per_well.items():
            well_dir = save_path / well_name
            well_dir.mkdir(parents=True, exist_ok=True)
            well_data["segment_df"].to_csv(
                well_dir / "segment_count_dataset.csv",
                index=False,
                encoding="utf-8-sig",
            )
        config_dict = {
            "exist_exp_dir": exist_exp_dir,
            "well_names": selected_wells,
            "prob_weight_gamma": prob_weight_gamma,
            "gt_min_points_per_segment": gt_min_points_per_segment,
            "gt_rounding_mode": gt_rounding_mode,
            "density_gt_col": density_gt_col,
            "default_feature_cols": DEFAULT_FEATURE_COLS,
        }
        with open(save_path / "config.json", "w", encoding="utf-8") as file_obj:
            json.dump(config_dict, file_obj, ensure_ascii=False, indent=2)

    return dataset_df, per_well, config


def parse_well_names(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exist-exp-dir", default=str(DEFAULT_EXIST_EXP_DIR))
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--well-names", default="")
    parser.add_argument("--prob-weight-gamma", type=float, default=2.0)
    parser.add_argument("--gt-min-points-per-segment", type=int, default=1)
    parser.add_argument("--gt-rounding-mode", default="round")
    parser.add_argument("--density-gt-col", default="P10")
    args = parser.parse_args()

    well_names = parse_well_names(args.well_names) if args.well_names else None
    dataset_df, _, _ = build_segment_dataset(
        exist_exp_dir=str(args.exist_exp_dir),
        well_names=well_names,
        save_dir=str(args.save_dir),
        prob_weight_gamma=float(args.prob_weight_gamma),
        gt_min_points_per_segment=int(args.gt_min_points_per_segment),
        gt_rounding_mode=str(args.gt_rounding_mode),
        density_gt_col=str(args.density_gt_col),
    )
    print(
        json.dumps(
            {
                "save_dir": str(args.save_dir),
                "num_segments": int(len(dataset_df)),
                "num_wells": int(dataset_df["WellName"].nunique()) if not dataset_df.empty else 0,
                "default_feature_cols": DEFAULT_FEATURE_COLS,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
