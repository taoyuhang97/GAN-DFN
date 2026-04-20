from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .runtime import load_target_sample_df, sanitize


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
                "AssignmentTopKMeanSimilarity": pd.to_numeric(best_row.get("TopKMeanSimilarity", np.nan), errors="coerce"),
                "AssignmentBestSimilarityScore": best_score,
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

    candidate_score_df = pd.DataFrame()
    strata_summary_df = pd.DataFrame()
    auto_assignment_df = pd.DataFrame()

    if np.any(auto_mask):
        if registry_df.empty:
            raise ValueError("Cannot auto determine strata because expert registry is empty")
        auto_input_df = resolved_df.loc[auto_mask].copy()
        interval_paths, _ = build_target_interval_files(
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
