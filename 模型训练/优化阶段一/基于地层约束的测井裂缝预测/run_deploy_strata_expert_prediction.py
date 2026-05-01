from strata_expert_deploy.pipeline import main as modular_main


if __name__ == "__main__":
    raise SystemExit(modular_main())

# Legacy monolithic implementation is kept below for reference.

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTER_FLOW_SCRIPT = ROOT / "run_outer_holdout_expert_validation.py"
NEAREST_SCRIPT_PATH = next(
    path for path in ROOT.rglob("nearest_imaging_well_predict_then_refine.py")
    if "_tmp_ascii_" not in str(path)
)
RAW_POINT_SCRIPT_PATH = next(
    path for path in ROOT.rglob("raw_point_guided_segment_refine.py")
    if "_tmp_ascii_" not in str(path)
)

FLOW_RESULT_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果"
)
DEFAULT_SAVE_ROOT = FLOW_RESULT_ROOT / "deploy_runs"
DEFAULT_LOG_FEATURES = ["AC", "GR"]
MODULE_CACHE: dict[str, object] = {}


def sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(text)).strip("_")


def parse_json_list(raw: str) -> list[str]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    except Exception:
        pass
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    return [
        item.strip().strip("'\"")
        for item in raw.split(",")
        if item.strip().strip("'\"")
    ]


def load_module(module_path: Path, module_name: str):
    cache_key = f"{module_name}:{module_path}"
    if cache_key in MODULE_CACHE:
        return MODULE_CACHE[cache_key]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    added_to_syspath = False
    parent_dir = str(module_path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        added_to_syspath = True
    try:
        spec.loader.exec_module(module)
    finally:
        if added_to_syspath:
            try:
                sys.path.remove(parent_dir)
            except ValueError:
                pass
    MODULE_CACHE[cache_key] = module
    return module


def resolve_depth_column(df: pd.DataFrame) -> str:
    for col in ["TVD", "DEPT", "MD"]:
        if col in df.columns:
            return col
    raise ValueError("No depth column found. Expected one of TVD/DEPT/MD.")


def normalize_target_strata_range_df(
    strata_range_csv: Path,
    target_well_name: str,
) -> pd.DataFrame:
    df = pd.read_csv(strata_range_csv, encoding="utf-8-sig")
    required = {"DepthMin", "DepthMax"}
    missing = [col for col in sorted(required) if col not in df.columns]
    if missing:
        raise ValueError(f"Target strata range CSV missing required columns: {missing}")

    out = df.copy()
    if "WellName" not in out.columns:
        out["WellName"] = str(target_well_name).strip()
    out["WellName"] = out["WellName"].astype(str).str.strip()
    out = out[out["WellName"] == str(target_well_name).strip()].copy()
    if out.empty:
        raise ValueError(f"No strata rows found for target well: {target_well_name}")

    if "StrataName" not in out.columns:
        out["StrataName"] = ""
    out["StrataName"] = out["StrataName"].fillna("").astype(str).str.strip()
    out["DepthMin"] = pd.to_numeric(out["DepthMin"], errors="coerce")
    out["DepthMax"] = pd.to_numeric(out["DepthMax"], errors="coerce")
    out = out[out["DepthMin"].notna() & out["DepthMax"].notna()].copy()
    if out.empty:
        raise ValueError("No valid target ranges remain after cleaning target strata range CSV")

    if "StrataTop" not in out.columns:
        out["StrataTop"] = np.nan
    if "StrataBase" not in out.columns:
        out["StrataBase"] = np.nan
    out["StrataTop"] = pd.to_numeric(out["StrataTop"], errors="coerce")
    out["StrataBase"] = pd.to_numeric(out["StrataBase"], errors="coerce")
    candidate_id_cols = [
        "RangeId",
        "IntervalId",
        "SegmentId",
        "RangeName",
        "IntervalName",
        "SegmentName",
    ]
    out = out.reset_index(drop=True)
    out["RangeOrder"] = np.arange(len(out), dtype=np.int64)
    out["ProvidedStrataName"] = out["StrataName"]
    out["RangeId"] = ""
    for col in candidate_id_cols:
        if col in out.columns:
            candidate = out[col].fillna("").astype(str).str.strip()
            fill_mask = out["RangeId"].eq("") & candidate.ne("")
            out.loc[fill_mask, "RangeId"] = candidate.loc[fill_mask]
    fill_mask = out["RangeId"].eq("")
    if fill_mask.any():
        out.loc[fill_mask, "RangeId"] = [
            f"interval_{int(idx) + 1:03d}"
            for idx in out.loc[fill_mask, "RangeOrder"].tolist()
        ]
    out["IntervalKey"] = [
        f"{int(order) + 1:03d}_{sanitize(range_id) or f'interval_{int(order) + 1:03d}'}"
        for order, range_id in zip(out["RangeOrder"].tolist(), out["RangeId"].tolist())
    ]
    return out.reset_index(drop=True)


def load_target_sample_df(
    target_sample_csv: Path,
    target_well_name: str,
) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(target_sample_csv, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Target sample CSV is empty: {target_sample_csv}")

    depth_col = resolve_depth_column(df)
    out_df = df.copy()
    out_df[depth_col] = pd.to_numeric(out_df[depth_col], errors="coerce")
    out_df = out_df[out_df[depth_col].notna()].copy()
    if "ROW_IN_WELL" in out_df.columns:
        out_df["ROW_IN_WELL"] = pd.to_numeric(out_df["ROW_IN_WELL"], errors="coerce")
        out_df = out_df.sort_values(["ROW_IN_WELL", depth_col], na_position="last").reset_index(drop=True)
    else:
        out_df = out_df.sort_values(depth_col).reset_index(drop=True)
        out_df["ROW_IN_WELL"] = np.arange(len(out_df), dtype=np.int64)
    out_df["WellName"] = str(target_well_name).strip()
    if "DepthForStrata" not in out_df.columns:
        out_df["DepthForStrata"] = out_df[depth_col]
    return out_df, depth_col


def discover_available_library_strata(
    stage1_library_dir: Path,
    stage2_library_dir: Path,
) -> list[str]:
    strata_names = []
    seen = set()
    for stage1_dir in sorted(stage1_library_dir.glob("*_lstm")):
        if not stage1_dir.is_dir():
            continue
        strata_name = str(stage1_dir.name)
        if strata_name.endswith("_lstm"):
            strata_name = strata_name[:-5]
        strata_name = str(strata_name).strip()
        if not strata_name or strata_name in seen:
            continue
        stage2_dir = stage2_library_dir / f"{sanitize(strata_name)}_refine"
        if not stage2_dir.exists():
            continue
        strata_names.append(strata_name)
        seen.add(strata_name)
    return strata_names


def build_target_interval_files(
    target_sample_csv: Path,
    target_well_name: str,
    target_range_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, Path], pd.DataFrame]:
    out_df, depth_col = load_target_sample_df(
        target_sample_csv=target_sample_csv,
        target_well_name=target_well_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    interval_paths: dict[str, Path] = {}
    plan_rows = []
    sort_cols = [col for col in ["RangeOrder", "DepthMin", "DepthMax"] if col in target_range_df.columns]
    work_df = target_range_df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else target_range_df.copy()
    for row in work_df.to_dict(orient="records"):
        interval_key = str(row.get("IntervalKey", "")).strip()
        if not interval_key:
            continue
        range_id = str(row.get("RangeId", interval_key)).strip()
        depth_min = float(min(float(row["DepthMin"]), float(row["DepthMax"])))
        depth_max = float(max(float(row["DepthMin"]), float(row["DepthMax"])))
        mask = out_df[depth_col].ge(depth_min - 1e-8) & out_df[depth_col].le(depth_max + 1e-8)
        interval_df = out_df.loc[mask].copy()
        interval_dir = output_dir / interval_key
        interval_dir.mkdir(parents=True, exist_ok=True)
        save_path = interval_dir / f"{target_well_name}_sample.csv"

        if not interval_df.empty:
            interval_df["InputRangeId"] = range_id
            interval_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            interval_paths[interval_key] = save_path

        plan_rows.append(
            {
                "WellName": target_well_name,
                "RangeId": range_id,
                "IntervalKey": interval_key,
                "RangeOrder": row.get("RangeOrder", np.nan),
                "ProvidedStrataName": str(row.get("ProvidedStrataName", "")).strip(),
                "DepthMin": depth_min,
                "DepthMax": depth_max,
                "TargetRows": int(len(interval_df)),
                "ExportStatus": "ok" if not interval_df.empty else "empty_after_depth_filter",
                "TargetSampleCsv": str(save_path),
            }
        )

    plan_df = pd.DataFrame(plan_rows)
    plan_df.to_csv(output_dir / "interval_export_plan.csv", index=False, encoding="utf-8-sig")
    return interval_paths, plan_df


def compute_interval_strata_scores(
    outer_module,
    interval_paths: dict[str, Path],
    interval_range_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    requested_log_features: list[str],
) -> pd.DataFrame:
    rows = []
    cache: dict[tuple[str, tuple[str, ...], tuple[str, ...]], pd.DataFrame] = {}
    registry_rows = registry_df.to_dict(orient="records")
    sort_cols = [col for col in ["RangeOrder", "DepthMin", "DepthMax"] if col in interval_range_df.columns]
    interval_rows = (
        interval_range_df.sort_values(sort_cols).to_dict(orient="records")
        if sort_cols
        else interval_range_df.to_dict(orient="records")
    )
    for interval_row in interval_rows:
        interval_key = str(interval_row.get("IntervalKey", "")).strip()
        target_csv = interval_paths.get(interval_key)
        if target_csv is None or not target_csv.exists():
            continue
        for registry_row in registry_rows:
            reference_sample_csv = Path(str(registry_row.get("reference_sample_csv", "")))
            stage1_config_path = Path(str(registry_row.get("stage1_config_path", "")))
            if not reference_sample_csv.exists() or not stage1_config_path.exists():
                continue
            try:
                feature_cols, feature_groups = outer_module.resolve_stage1_feature_groups(
                    stage1_config_path,
                    requested_log_features,
                )
            except Exception:
                continue
            if not feature_cols:
                continue
            log_features = [col for col in requested_log_features if col in feature_cols]
            target_key = (str(target_csv), tuple(feature_cols), tuple(log_features))
            if target_key not in cache:
                cache[target_key] = outer_module.prepare_similarity_df(
                    target_csv,
                    feature_cols,
                    log_features,
                )
            reference_key = (str(reference_sample_csv), tuple(feature_cols), tuple(log_features))
            if reference_key not in cache:
                cache[reference_key] = outer_module.prepare_similarity_df(
                    reference_sample_csv,
                    feature_cols,
                    log_features,
                )

            target_df = cache[target_key]
            candidate_df = cache[reference_key]
            group_scores = {
                group_name: outer_module.compute_group_distance(target_df, candidate_df, cols)
                for group_name, cols in feature_groups.items()
            }
            weighted_sum = 0.0
            weight_sum = 0.0
            for group_name, weight in outer_module.SIMILARITY_GROUP_WEIGHTS.items():
                group_value = group_scores.get(group_name, np.nan)
                if np.isfinite(group_value):
                    weighted_sum += float(weight) * float(group_value)
                    weight_sum += float(weight)

            rows.append(
                {
                    "IntervalKey": interval_key,
                    "RangeId": str(interval_row.get("RangeId", "")).strip(),
                    "RangeOrder": interval_row.get("RangeOrder", np.nan),
                    "DepthMin": pd.to_numeric(interval_row.get("DepthMin", np.nan), errors="coerce"),
                    "DepthMax": pd.to_numeric(interval_row.get("DepthMax", np.nan), errors="coerce"),
                    "ProvidedStrataName": str(interval_row.get("ProvidedStrataName", "")).strip(),
                    "CandidateStrataName": str(registry_row.get("strata_name", "")).strip(),
                    "CandidateExpertWell": str(registry_row.get("expert_well", "")).strip(),
                    "similarity_score": float(weighted_sum / weight_sum) if weight_sum > 0 else np.nan,
                    "similarity_score_seismic": group_scores.get("seismic", np.nan),
                    "similarity_score_AC": group_scores.get("AC", np.nan),
                    "similarity_score_GR": group_scores.get("GR", np.nan),
                    "inner_stage1_f1": pd.to_numeric(registry_row.get("inner_stage1_f1", np.nan), errors="coerce"),
                    "inner_stage2_count_error_pct": pd.to_numeric(
                        registry_row.get("inner_stage2_count_error_pct", np.nan),
                        errors="coerce",
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_interval_strata_scores(
    score_df: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if score_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work_df = score_df.copy()
    work_df["similarity_score"] = pd.to_numeric(work_df["similarity_score"], errors="coerce")
    work_df["inner_stage1_f1"] = pd.to_numeric(work_df["inner_stage1_f1"], errors="coerce")
    work_df["inner_stage2_count_error_pct"] = pd.to_numeric(
        work_df["inner_stage2_count_error_pct"],
        errors="coerce",
    )

    summary_rows = []
    for (interval_key, strata_name), sub_df in work_df.groupby(["IntervalKey", "CandidateStrataName"], sort=False):
        valid_sub = sub_df[sub_df["similarity_score"].notna()].copy()
        if valid_sub.empty:
            continue
        valid_sub = valid_sub.sort_values(
            ["similarity_score", "inner_stage1_f1", "inner_stage2_count_error_pct", "CandidateExpertWell"],
            ascending=[True, False, True, True],
        ).reset_index(drop=True)
        best_row = valid_sub.iloc[0]
        top_n = max(1, min(int(top_k), int(len(valid_sub))))
        top_scores = valid_sub["similarity_score"].iloc[:top_n].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "IntervalKey": interval_key,
                "RangeId": str(best_row.get("RangeId", "")).strip(),
                "RangeOrder": pd.to_numeric(best_row.get("RangeOrder", np.nan), errors="coerce"),
                "DepthMin": pd.to_numeric(best_row.get("DepthMin", np.nan), errors="coerce"),
                "DepthMax": pd.to_numeric(best_row.get("DepthMax", np.nan), errors="coerce"),
                "ProvidedStrataName": str(best_row.get("ProvidedStrataName", "")).strip(),
                "CandidateStrataName": str(strata_name).strip(),
                "NumExpertCandidates": int(len(valid_sub)),
                "TopKUsed": int(top_n),
                "TopKMeanSimilarity": float(np.nanmean(top_scores)),
                "BestSimilarityScore": float(best_row["similarity_score"]),
                "BestExpertWell": str(best_row.get("CandidateExpertWell", "")).strip(),
                "BestStage1F1": pd.to_numeric(best_row.get("inner_stage1_f1", np.nan), errors="coerce"),
                "BestStage2CountErrorPct": pd.to_numeric(
                    best_row.get("inner_stage2_count_error_pct", np.nan),
                    errors="coerce",
                ),
            }
        )

    strata_summary_df = pd.DataFrame(summary_rows)
    if strata_summary_df.empty:
        return strata_summary_df, pd.DataFrame()

    assignment_rows = []
    for interval_key, sub_df in strata_summary_df.groupby("IntervalKey", sort=False):
        ranked_df = sub_df.sort_values(
            ["BestSimilarityScore", "TopKMeanSimilarity", "BestStage1F1", "BestStage2CountErrorPct", "CandidateStrataName"],
            ascending=[True, True, False, True, True],
        ).reset_index(drop=True)
        best_row = ranked_df.iloc[0]
        second_score = pd.to_numeric(ranked_df.iloc[1]["BestSimilarityScore"], errors="coerce") if len(ranked_df) > 1 else np.nan
        best_score = pd.to_numeric(best_row["BestSimilarityScore"], errors="coerce")
        assignment_rows.append(
            {
                "IntervalKey": interval_key,
                "RangeId": str(best_row.get("RangeId", "")).strip(),
                "RangeOrder": pd.to_numeric(best_row.get("RangeOrder", np.nan), errors="coerce"),
                "DepthMin": pd.to_numeric(best_row.get("DepthMin", np.nan), errors="coerce"),
                "DepthMax": pd.to_numeric(best_row.get("DepthMax", np.nan), errors="coerce"),
                "ProvidedStrataName": str(best_row.get("ProvidedStrataName", "")).strip(),
                "AssignedStrataName": str(best_row.get("CandidateStrataName", "")).strip(),
                "AssignmentTopKMeanSimilarity": best_score,
                "AssignmentBestSimilarityScore": pd.to_numeric(best_row.get("BestSimilarityScore", np.nan), errors="coerce"),
                "AssignmentScoreMargin": second_score - best_score if np.isfinite(second_score) and np.isfinite(best_score) else np.nan,
                "AssignedBestExpertWell": str(best_row.get("BestExpertWell", "")).strip(),
            }
        )
    return strata_summary_df, pd.DataFrame(assignment_rows)


def resolve_target_strata_range_df(
    outer_module,
    target_sample_csv: Path,
    target_well_name: str,
    target_range_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    requested_log_features: list[str],
    output_dir: Path,
    determination_mode: str,
    determination_top_k: int,
) -> tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_df = target_range_df.copy().reset_index(drop=True)
    normalized_df.to_csv(output_dir / "target_range_input_normalized.csv", index=False, encoding="utf-8-sig")

    mode = str(determination_mode or "auto_if_missing").strip().lower()
    allowed_modes = {"use_csv", "auto_if_missing", "always_auto"}
    if mode not in allowed_modes:
        raise ValueError(f"Unsupported strata_determination_mode: {determination_mode}, allowed={sorted(allowed_modes)}")

    resolved_df = normalized_df.copy()
    resolved_df["StrataName"] = resolved_df["ProvidedStrataName"].fillna("").astype(str).str.strip()
    resolved_df["StrataAssignmentSource"] = np.where(resolved_df["StrataName"].ne(""), "provided", "")
    resolved_df["AssignmentTopKMeanSimilarity"] = np.nan
    resolved_df["AssignmentBestSimilarityScore"] = np.nan
    resolved_df["AssignmentScoreMargin"] = np.nan
    resolved_df["AssignedBestExpertWell"] = ""

    if mode == "use_csv":
        if resolved_df["StrataName"].eq("").any():
            raise ValueError("target_strata_range_csv contains missing StrataName rows while strata_determination_mode=use_csv")
        resolved_df.to_csv(output_dir / "resolved_target_strata_range.csv", index=False, encoding="utf-8-sig")
        return resolved_df, {
            "mode": mode,
            "num_total_ranges": int(len(resolved_df)),
            "num_auto_assigned_ranges": 0,
            "num_provided_ranges": int(len(resolved_df)),
            "resolved_csv": str(output_dir / "resolved_target_strata_range.csv"),
        }

    auto_mask = resolved_df["StrataName"].eq("")
    if mode == "always_auto":
        auto_mask = np.ones(len(resolved_df), dtype=bool)

    interval_plan_df = pd.DataFrame()
    candidate_score_df = pd.DataFrame()
    strata_summary_df = pd.DataFrame()
    auto_assignment_df = pd.DataFrame()

    if np.any(auto_mask):
        if registry_df.empty:
            raise ValueError("Cannot auto determine strata because expert registry is empty")
        auto_input_df = resolved_df.loc[auto_mask].copy()
        interval_paths, interval_plan_df = build_target_interval_files(
            target_sample_csv=target_sample_csv,
            target_well_name=target_well_name,
            target_range_df=auto_input_df,
            output_dir=output_dir / "interval_samples",
        )
        candidate_score_df = compute_interval_strata_scores(
            outer_module=outer_module,
            interval_paths=interval_paths,
            interval_range_df=auto_input_df,
            registry_df=registry_df,
            requested_log_features=requested_log_features,
        )
        candidate_score_df.to_csv(output_dir / "auto_interval_candidate_scores.csv", index=False, encoding="utf-8-sig")
        strata_summary_df, auto_assignment_df = summarize_interval_strata_scores(
            score_df=candidate_score_df,
            top_k=int(determination_top_k),
        )
        strata_summary_df.to_csv(output_dir / "auto_interval_strata_summary.csv", index=False, encoding="utf-8-sig")
        auto_assignment_df.to_csv(output_dir / "auto_interval_strata_assignment.csv", index=False, encoding="utf-8-sig")
        if auto_assignment_df.empty:
            raise ValueError("Failed to auto determine strata for target ranges: no valid assignment rows were produced")

        assignment_lookup = auto_assignment_df.set_index("IntervalKey")
        for col in [
            "AssignedStrataName",
            "AssignmentTopKMeanSimilarity",
            "AssignmentBestSimilarityScore",
            "AssignmentScoreMargin",
            "AssignedBestExpertWell",
        ]:
            resolved_df.loc[auto_mask, col] = resolved_df.loc[auto_mask, "IntervalKey"].map(assignment_lookup[col])
        missing_assigned_mask = auto_mask & resolved_df["AssignedStrataName"].fillna("").astype(str).str.strip().eq("")
        if np.any(missing_assigned_mask):
            unresolved_range_ids = resolved_df.loc[missing_assigned_mask, "RangeId"].astype(str).tolist()
            raise ValueError(f"Failed to auto determine strata for ranges: {unresolved_range_ids}")
        resolved_df.loc[auto_mask, "StrataName"] = (
            resolved_df.loc[auto_mask, "AssignedStrataName"].fillna("").astype(str).str.strip()
        )
        resolved_df.loc[auto_mask, "StrataAssignmentSource"] = "auto_similarity"

    resolved_df["StrataName"] = resolved_df["StrataName"].fillna("").astype(str).str.strip()
    if resolved_df["StrataName"].eq("").any():
        raise ValueError("Resolved target ranges still contain empty StrataName values")

    assignment_export_df = resolved_df[
        [
            "WellName",
            "RangeId",
            "IntervalKey",
            "RangeOrder",
            "DepthMin",
            "DepthMax",
            "ProvidedStrataName",
            "StrataName",
            "StrataAssignmentSource",
            "AssignmentTopKMeanSimilarity",
            "AssignmentBestSimilarityScore",
            "AssignmentScoreMargin",
            "AssignedBestExpertWell",
        ]
    ].copy()
    assignment_export_df = assignment_export_df.sort_values(["RangeOrder", "DepthMin", "DepthMax"]).reset_index(drop=True)
    assignment_export_df.to_csv(output_dir / "resolved_interval_strata_assignment.csv", index=False, encoding="utf-8-sig")
    resolved_df.to_csv(output_dir / "resolved_target_strata_range.csv", index=False, encoding="utf-8-sig")
    return resolved_df.reset_index(drop=True), {
        "mode": mode,
        "num_total_ranges": int(len(resolved_df)),
        "num_auto_assigned_ranges": int(np.sum(auto_mask.astype(np.int64))),
        "num_provided_ranges": int(np.sum((~auto_mask).astype(np.int64))),
        "normalized_csv": str(output_dir / "target_range_input_normalized.csv"),
        "resolved_csv": str(output_dir / "resolved_target_strata_range.csv"),
        "assignment_csv": str(output_dir / "resolved_interval_strata_assignment.csv"),
        "candidate_score_csv": str(output_dir / "auto_interval_candidate_scores.csv"),
        "summary_score_csv": str(output_dir / "auto_interval_strata_summary.csv"),
    }


def build_target_segment_files(
    target_sample_csv: Path,
    target_well_name: str,
    target_range_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]:
    out_df, depth_col = load_target_sample_df(
        target_sample_csv=target_sample_csv,
        target_well_name=target_well_name,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    segment_paths: dict[str, Path] = {}
    range_rows = []
    plan_rows = []

    grouped = target_range_df.groupby("StrataName", sort=False)
    for strata_name, sub_range in grouped:
        depth_min = float(min(sub_range["DepthMin"].min(), sub_range["DepthMax"].min()))
        depth_max = float(max(sub_range["DepthMin"].max(), sub_range["DepthMax"].max()))
        mask = np.zeros(len(out_df), dtype=bool)
        for row in sub_range.to_dict(orient="records"):
            interval_min = float(min(float(row["DepthMin"]), float(row["DepthMax"])))
            interval_max = float(max(float(row["DepthMin"]), float(row["DepthMax"])))
            mask |= out_df[depth_col].ge(interval_min - 1e-8) & out_df[depth_col].le(interval_max + 1e-8)
        strata_df = out_df.loc[mask].copy()
        strata_dir = output_dir / sanitize(strata_name)
        strata_dir.mkdir(parents=True, exist_ok=True)
        save_path = strata_dir / f"{target_well_name}_sample.csv"

        if strata_df.empty:
            plan_rows.append(
                {
                    "WellName": target_well_name,
                    "StrataName": strata_name,
                    "DepthMin": depth_min,
                    "DepthMax": depth_max,
                    "TargetRows": 0,
                    "ExportStatus": "empty_after_depth_filter",
                    "RangeMergeCount": int(len(sub_range)),
                    "TargetSampleCsv": str(save_path),
                }
            )
            continue

        strata_df["StrataName"] = str(strata_name).strip()
        strata_df["StrataTop"] = float(sub_range["StrataTop"].dropna().iloc[0]) if sub_range["StrataTop"].notna().any() else np.nan
        strata_df["StrataBase"] = float(sub_range["StrataBase"].dropna().iloc[0]) if sub_range["StrataBase"].notna().any() else np.nan
        strata_df.to_csv(save_path, index=False, encoding="utf-8-sig")

        segment_paths[strata_name] = save_path
        range_rows.append(
            {
                "WellName": target_well_name,
                "StrataName": strata_name,
                "DepthMin": depth_min,
                "DepthMax": depth_max,
                "StrataTop": float(sub_range["StrataTop"].dropna().iloc[0]) if sub_range["StrataTop"].notna().any() else np.nan,
                "StrataBase": float(sub_range["StrataBase"].dropna().iloc[0]) if sub_range["StrataBase"].notna().any() else np.nan,
            }
        )
        plan_rows.append(
            {
                "WellName": target_well_name,
                "StrataName": strata_name,
                "DepthMin": depth_min,
                "DepthMax": depth_max,
                "TargetRows": int(len(strata_df)),
                "ExportStatus": "ok",
                "RangeMergeCount": int(len(sub_range)),
                "TargetSampleCsv": str(save_path),
            }
        )

    merged_range_df = pd.DataFrame(range_rows)
    merged_range_df.to_csv(output_dir / "target_strata_range_merged.csv", index=False, encoding="utf-8-sig")
    plan_df = pd.DataFrame(plan_rows)
    plan_df.to_csv(output_dir / "target_segment_export_plan.csv", index=False, encoding="utf-8-sig")
    return segment_paths, merged_range_df, plan_df


def run_second_stage_prediction(
    raw_module,
    target_well_name: str,
    strata_name: str,
    stage1_pred_csv: Path,
    selected_expert_row: dict,
    strata_range_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = raw_module.load_model_artifact(Path(str(selected_expert_row["stage2_model_dir"])))
    config = raw_module.dataset_builder.make_config(
        exist_exp_dir=str(artifact.get("exist_exp_dir", "")),
        prob_weight_gamma=float(artifact.get("prob_weight_gamma", 2.0)),
        gt_min_points_per_segment=1,
        gt_rounding_mode="round",
        density_gt_col="P10",
    )
    boundary_expand_mode = raw_module.validate_boundary_expand_mode(
        str(artifact.get("boundary_expand_mode", "none"))
    )
    boundary_expand_prob_min = float(artifact.get("boundary_expand_prob_min", 0.35))
    boundary_expand_max_steps = int(artifact.get("boundary_expand_max_steps", 0))
    boundary_expand_max_depth = float(artifact.get("boundary_expand_max_depth", 0.0))

    well_data = raw_module.build_predict_segment_dataset(
        exist_csv=str(stage1_pred_csv),
        well_name=target_well_name,
        config=config,
        strata_name=strata_name,
        strata_range_df=strata_range_df,
        boundary_expand_mode=boundary_expand_mode,
        boundary_expand_prob_min=boundary_expand_prob_min,
        boundary_expand_max_steps=boundary_expand_max_steps,
        boundary_expand_max_depth=boundary_expand_max_depth,
    )
    if well_data["segment_df"].empty:
        pred_segment_df = pd.DataFrame()
        pred_points_df = pd.DataFrame()
        lowconf_mask = np.array([], dtype=bool)
    else:
        pred_segment_df, lowconf_mask = raw_module.predict_segments_with_artifact(
            segment_df=well_data["segment_df"].copy(),
            artifact=artifact,
            include_gt_columns=False,
        )
        pred_points_df = raw_module.build_pred_points_from_segment_df(
            df_sorted=well_data["df_sorted"],
            payloads=well_data["payloads"],
            pred_segment_df=pred_segment_df,
            config=config,
            orientation_strategy=dict(artifact.get("orientation_strategy") or {}),
        )

    pred_segment_df.to_csv(output_dir / "pred_segment_summary.csv", index=False, encoding="utf-8-sig")
    pred_points_df.to_csv(output_dir / "pred_fracture_points.csv", index=False, encoding="utf-8-sig")
    if "df" in well_data and isinstance(well_data["df"], pd.DataFrame):
        well_data["df"].to_csv(output_dir / "input_predict_log.csv", index=False, encoding="utf-8-sig")

    info = {
        "stage2_model_dir": str(selected_expert_row["stage2_model_dir"]),
        "artifact_target": str(artifact.get("artifact_target", "")),
        "artifact_type": str(artifact.get("artifact_type", "")),
        "orientation_mode": str(artifact.get("orientation_strategy", {}).get("mode", "none")),
        "orientation_num_families": int(artifact.get("orientation_strategy", {}).get("num_families_fitted", 0)),
        "num_segments": int(len(pred_segment_df)),
        "num_lowconf_segments": int(np.sum(lowconf_mask.astype(np.int64))),
        "num_pred_points": int(len(pred_points_df)),
    }
    return pred_segment_df, pred_points_df, info


def sort_by_depth_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["TVD", "DEPT", "MD", "Depth", "DepthForStrata"]:
        if col in df.columns:
            out = df.copy()
            out[col] = pd.to_numeric(out[col], errors="coerce")
            return out.sort_values(col, na_position="last").reset_index(drop=True)
    return df.reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--target-sample-csv", required=True)
    parser.add_argument("--target-well-name", required=True)
    parser.add_argument("--target-strata-range-csv", required=True)
    parser.add_argument("--stage1-library-dir", required=True)
    parser.add_argument("--stage2-library-dir", required=True)
    parser.add_argument("--requested-log-features-json", default=json.dumps(DEFAULT_LOG_FEATURES, ensure_ascii=False))
    parser.add_argument(
        "--strata-determination-mode",
        default="auto_if_missing",
        choices=["use_csv", "auto_if_missing", "always_auto"],
    )
    parser.add_argument("--strata-determination-topk", type=int, default=2)
    parser.add_argument("--unknown-strata-policy", default="skip", choices=["skip", "error"])
    parser.add_argument("--first-stage-threshold", type=float, default=np.nan)
    parser.add_argument("--result-dir", default="")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    target_sample_csv = Path(args.target_sample_csv)
    target_range_csv = Path(args.target_strata_range_csv)
    stage1_library_dir = Path(args.stage1_library_dir)
    stage2_library_dir = Path(args.stage2_library_dir)
    if not target_sample_csv.exists():
        raise FileNotFoundError(f"Target sample CSV not found: {target_sample_csv}")
    if not target_range_csv.exists():
        raise FileNotFoundError(f"Target strata range CSV not found: {target_range_csv}")
    if not stage1_library_dir.exists():
        raise FileNotFoundError(f"Stage1 library dir not found: {stage1_library_dir}")
    if not stage2_library_dir.exists():
        raise FileNotFoundError(f"Stage2 library dir not found: {stage2_library_dir}")

    requested_log_features = parse_json_list(args.requested_log_features_json)
    if not requested_log_features:
        raise ValueError("requested_log_features cannot be empty")

    result_root = Path(args.result_dir) if str(args.result_dir).strip() else (DEFAULT_SAVE_ROOT / sanitize(args.exp_id))
    result_root.mkdir(parents=True, exist_ok=True)

    outer_module = load_module(OUTER_FLOW_SCRIPT, "deploy_outer_flow_module")
    nearest_module = load_module(NEAREST_SCRIPT_PATH, "deploy_nearest_module")
    raw_module = load_module(RAW_POINT_SCRIPT_PATH, "deploy_raw_point_module")

    available_library_strata = discover_available_library_strata(
        stage1_library_dir=stage1_library_dir,
        stage2_library_dir=stage2_library_dir,
    )
    if not available_library_strata:
        raise ValueError("No overlapping strata directories were found between stage1_library_dir and stage2_library_dir")
    registry_df = outer_module.build_expert_registry(
        stage1_library_dir=stage1_library_dir,
        stage2_library_dir=stage2_library_dir,
        target_strata=available_library_strata,
    )
    registry_df.to_csv(result_root / "expert_model_registry.csv", index=False, encoding="utf-8-sig")

    target_range_input_df = normalize_target_strata_range_df(
        strata_range_csv=target_range_csv,
        target_well_name=str(args.target_well_name),
    )
    target_range_df, strata_resolution_info = resolve_target_strata_range_df(
        outer_module=outer_module,
        target_sample_csv=target_sample_csv,
        target_well_name=str(args.target_well_name),
        target_range_df=target_range_input_df,
        registry_df=registry_df,
        requested_log_features=requested_log_features,
        output_dir=result_root / "target_range_resolution",
        determination_mode=str(args.strata_determination_mode),
        determination_top_k=int(args.strata_determination_topk),
    )
    segment_paths, merged_range_df, target_segment_plan_df = build_target_segment_files(
        target_sample_csv=target_sample_csv,
        target_well_name=str(args.target_well_name),
        target_range_df=target_range_df,
        output_dir=result_root / "target_segments",
    )
    if not segment_paths:
        raise ValueError("No target strata segment files were exported")

    target_strata = list(segment_paths.keys())

    available_strata = set(registry_df["strata_name"].astype(str).tolist()) if not registry_df.empty else set()
    unknown_rows = []
    known_strata = []
    for strata_name in target_strata:
        if strata_name in available_strata:
            known_strata.append(strata_name)
        else:
            unknown_rows.append(
                {
                    "WellName": args.target_well_name,
                    "StrataName": strata_name,
                    "Status": "missing_library_strata",
                }
            )
    if unknown_rows and args.unknown_strata_policy == "error":
        raise ValueError(f"Missing expert library strata: {[row['StrataName'] for row in unknown_rows]}")

    known_segment_paths = {name: path for name, path in segment_paths.items() if name in known_strata}
    if not known_segment_paths:
        raise ValueError("No known strata remain after applying unknown_strata_policy")

    known_registry_df = registry_df[registry_df["strata_name"].astype(str).isin(known_strata)].copy()
    score_df = outer_module.compute_similarity_scores(
        segment_paths=known_segment_paths,
        registry_df=known_registry_df,
        requested_log_features=requested_log_features,
    )
    if score_df.empty:
        raise ValueError("No expert similarity rows were produced for known strata")
    score_df.to_csv(result_root / "expert_selection_scores.csv", index=False, encoding="utf-8-sig")

    selected_expert_df = outer_module.select_expert_for_strata(score_df)
    selected_expert_df.to_csv(result_root / "selected_expert_by_strata.csv", index=False, encoding="utf-8-sig")
    selected_map = {
        str(row["strata_name"]): row
        for row in selected_expert_df.to_dict(orient="records")
    }

    strata_status_rows = []
    stage1_frames = []
    segment_frames = []
    point_frames = []
    stage1_summary_rows = []
    stage2_summary_rows = []

    threshold_override = None if not np.isfinite(args.first_stage_threshold) else float(args.first_stage_threshold)
    for strata_name in target_strata:
        strata_output_dir = result_root / "prediction_by_strata" / sanitize(strata_name)
        strata_output_dir.mkdir(parents=True, exist_ok=True)

        if strata_name not in selected_map:
            strata_status_rows.append(
                {
                    "WellName": args.target_well_name,
                    "StrataName": strata_name,
                    "Status": "skipped_missing_selected_expert",
                    "SelectedExpertWell": "",
                    "TargetSampleCsv": str(segment_paths[strata_name]),
                }
            )
            continue

        selected_row = selected_map[strata_name]
        expert_well = str(selected_row["selected_expert_well"])
        stage1_info = nearest_module.run_first_stage_prediction(
            input_csv=segment_paths[strata_name],
            well_name=str(args.target_well_name),
            model_well=expert_well,
            model_dir=Path(str(selected_row["stage1_model_dir"])),
            output_dir=strata_output_dir,
            threshold_override=threshold_override,
        )
        stage1_df = pd.read_csv(stage1_info["output_csv"], encoding="utf-8-sig")
        stage1_df["StrataName"] = strata_name
        stage1_frames.append(stage1_df)
        stage1_summary_rows.append(
            {
                "WellName": args.target_well_name,
                "StrataName": strata_name,
                "SelectedExpertWell": expert_well,
                "Stage1ModelDir": str(selected_row["stage1_model_dir"]),
                "Stage1Threshold": stage1_info["threshold"],
                "Stage1NumInputRows": stage1_info["num_input_rows"],
                "Stage1NumRowsAfterClean": stage1_info["num_rows_after_clean"],
                "Stage1NumRemovedRows": stage1_info["num_removed_rows"],
                "Stage1NumSequences": stage1_info["num_sequences"],
                "Stage1NumPredictedPositiveCenters": stage1_info["num_predicted_positive_centers"],
                "Stage1OutputCsv": str(stage1_info["output_csv"]),
            }
        )

        pred_segment_df, pred_points_df, stage2_info = run_second_stage_prediction(
            raw_module=raw_module,
            target_well_name=str(args.target_well_name),
            strata_name=strata_name,
            stage1_pred_csv=Path(stage1_info["output_csv"]),
            selected_expert_row=selected_row,
            strata_range_df=merged_range_df,
            output_dir=strata_output_dir,
        )
        if not pred_segment_df.empty:
            pred_segment_df["StrataName"] = strata_name
            segment_frames.append(pred_segment_df)
        if not pred_points_df.empty:
            pred_points_df["StrataName"] = strata_name
            point_frames.append(pred_points_df)
        stage2_summary_rows.append(
            {
                "WellName": args.target_well_name,
                "StrataName": strata_name,
                "SelectedExpertWell": expert_well,
                "Stage2ModelDir": stage2_info["stage2_model_dir"],
                "ArtifactTarget": stage2_info["artifact_target"],
                "ArtifactType": stage2_info["artifact_type"],
                "OrientationMode": stage2_info["orientation_mode"],
                "OrientationNumFamilies": stage2_info["orientation_num_families"],
                "NumPredSegments": stage2_info["num_segments"],
                "NumLowconfSegments": stage2_info["num_lowconf_segments"],
                "NumPredPoints": stage2_info["num_pred_points"],
            }
        )
        strata_status_rows.append(
            {
                "WellName": args.target_well_name,
                "StrataName": strata_name,
                "Status": "ok",
                "SelectedExpertWell": expert_well,
                "TargetSampleCsv": str(segment_paths[strata_name]),
            }
        )

    strata_status_df = pd.DataFrame(strata_status_rows + unknown_rows)
    strata_status_df.to_csv(result_root / "deploy_strata_status.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stage1_summary_rows).to_csv(result_root / "stage1_deploy_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stage2_summary_rows).to_csv(result_root / "stage2_deploy_summary.csv", index=False, encoding="utf-8-sig")

    whole_stage1_df = sort_by_depth_if_possible(pd.concat(stage1_frames, ignore_index=True)) if stage1_frames else pd.DataFrame()
    whole_segment_df = sort_by_depth_if_possible(pd.concat(segment_frames, ignore_index=True)) if segment_frames else pd.DataFrame()
    whole_points_df = sort_by_depth_if_possible(pd.concat(point_frames, ignore_index=True)) if point_frames else pd.DataFrame()
    whole_stage1_df.to_csv(result_root / "whole_well_stage1_pred_full_log.csv", index=False, encoding="utf-8-sig")
    whole_segment_df.to_csv(result_root / "whole_well_pred_segment_summary.csv", index=False, encoding="utf-8-sig")
    whole_points_df.to_csv(result_root / "whole_well_pred_fracture_points.csv", index=False, encoding="utf-8-sig")

    summary = {
        "exp_id": args.exp_id,
        "target_well_name": str(args.target_well_name),
        "target_sample_csv": str(target_sample_csv),
        "target_strata_range_csv": str(target_range_csv),
        "strata_determination_mode": str(args.strata_determination_mode),
        "strata_determination_topk": int(args.strata_determination_topk),
        "strata_resolution": strata_resolution_info,
        "stage1_library_dir": str(stage1_library_dir),
        "stage2_library_dir": str(stage2_library_dir),
        "requested_log_features": requested_log_features,
        "unknown_strata_policy": str(args.unknown_strata_policy),
        "num_target_strata": int(len(target_strata)),
        "num_known_strata": int(len(known_strata)),
        "num_skipped_strata": int(len(strata_status_df[strata_status_df["Status"].astype(str).str.startswith("skipped") | strata_status_df["Status"].astype(str).eq("missing_library_strata")])),
        "whole_well_stage1_pred_csv": str(result_root / "whole_well_stage1_pred_full_log.csv"),
        "whole_well_segment_csv": str(result_root / "whole_well_pred_segment_summary.csv"),
        "whole_well_points_csv": str(result_root / "whole_well_pred_fracture_points.csv"),
        "status_csv": str(result_root / "deploy_strata_status.csv"),
        "selected_expert_csv": str(result_root / "selected_expert_by_strata.csv"),
        "score_csv": str(result_root / "expert_selection_scores.csv"),
        "result_root": str(result_root),
    }
    with (result_root / "deployment_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
