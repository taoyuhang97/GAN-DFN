from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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
