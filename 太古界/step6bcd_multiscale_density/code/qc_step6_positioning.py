from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_3d_density_sgy.json"
DEFAULT_MULTISCALE_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_multiscale_density_v1.json"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "output/candidate_cheye1_step6_positioning_qc_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step6 positioning QC and build a decision report.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--multiscale-config", type=Path, default=DEFAULT_MULTISCALE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-sub-qc", action="store_true", help="Only rebuild decision/report from existing sub-QC outputs.")
    parser.add_argument("--max-well-correlation-rows", type=int, default=1000000)
    parser.add_argument("--correlation-sample-size", type=int, default=1000000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(cmd: list[str], cwd: Path) -> None:
    print("[step6-positioning]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def get_nested(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def corr_value(payload: dict[str, Any], group: str, target: str, attr: str, method: str = "pearson") -> float | None:
    value = get_nested(payload, ["correlations", group, target, attr, method])
    return float(value) if isinstance(value, int | float) else None


def overlap_enrichment(payload: dict[str, Any], density_label: str, top_bin: str, mask_name: str) -> float | None:
    value = get_nested(payload, [density_label, "top_quantiles", top_bin, "overlap", mask_name, "enrichment"])
    return float(value) if isinstance(value, int | float) else None


def model_validation_summary(step6_summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layer, item in dict(step6_summary.get("model_summary", {})).items():
        out[layer] = {
            "train_rows": item.get("train_rows"),
            "validation_rows": item.get("validation_rows"),
            "validation_mae": item.get("validation_mae"),
            "validation_r2": item.get("validation_r2"),
            "validation_prediction_stats": item.get("validation_prediction_stats"),
        }
    return out


def build_decision(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    input_qc_path = output_dir / "input_alignment_qc" / "input_qc_summary.json"
    density_qc_dir = output_dir / "density_attribute_qc"
    density_qc_path = density_qc_dir / "step6_multiscale_qc_current.json"
    well_corr_path = density_qc_dir / "well_density_attribute_correlation.json"
    overlap_path = density_qc_dir / "top_density_attribute_overlap.json"
    lowcoh_path = density_qc_dir / "lowcoh_component_qc.json"
    step6_summary_path = Path(config["output_dir"]).resolve() / f"{config['target_block']['name']}_3d_density_sgy_summary.json"

    required = [input_qc_path, density_qc_path, well_corr_path, overlap_path, lowcoh_path, step6_summary_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing QC inputs: {missing}")

    input_qc = read_json(input_qc_path)
    density_qc = read_json(density_qc_path)
    well_corr = read_json(well_corr_path)
    overlap = read_json(overlap_path)
    lowcoh = read_json(lowcoh_path)
    step6_summary = read_json(step6_summary_path)

    scan = dict(step6_summary.get("training_scan_summary", {}))
    source_counts = dict(scan.get("source_kind_counts_seen", {}))
    real_rows = float(source_counts.get("real_well", 0.0) or 0.0)
    virtual_rows = float(source_counts.get("virtual_well", 0.0) or 0.0)
    virtual_to_real_row_ratio = virtual_rows / real_rows if real_rows > 0 else None

    target_counts = dict(well_corr.get("counts", {}).get("source_kind_counts", {}))
    target_weights = dict(well_corr.get("counts", {}).get("nominal_training_weight_sum_by_source_kind", {}))
    target_real_rows = float(target_counts.get("real_well", 0.0) or 0.0)
    target_virtual_rows = float(target_counts.get("virtual_well", 0.0) or 0.0)
    target_real_weight = float(target_weights.get("real_well", 0.0) or 0.0)
    target_virtual_weight = float(target_weights.get("virtual_well", 0.0) or 0.0)

    real_density_coh = corr_value(well_corr, "SourceKind=real_well", "DensityLabel", "Coherence")
    real_density_ant = corr_value(well_corr, "SourceKind=real_well", "DensityLabel", "AntTrack")
    real_flag_coh = corr_value(well_corr, "SourceKind=real_well", "GT_POINT_FLAG", "Coherence")
    real_flag_ant = corr_value(well_corr, "SourceKind=real_well", "GT_POINT_FLAG", "AntTrack")
    all_density_coh = corr_value(well_corr, "all", "DensityLabel", "Coherence")
    all_density_ant = corr_value(well_corr, "all", "DensityLabel", "AntTrack")

    original_top1_lowcoh = overlap_enrichment(overlap, "original_step6b", "top_1pct", "coherence_le_q10")
    original_top1_ant = overlap_enrichment(overlap, "original_step6b", "top_1pct", "anttrack_ge_q90")
    original_top1_curvmax = overlap_enrichment(overlap, "original_step6b", "top_1pct", "curvaturemax_ge_q90")
    original_top1_curvpos = overlap_enrichment(overlap, "original_step6b", "top_1pct", "curvaturepos_ge_q90")
    lowcoh_steep_top1_lowcoh = overlap_enrichment(overlap, "lowcoh_steep", "top_1pct", "coherence_le_q10")

    density_attr_corr = dict(density_qc.get("density_attribute_correlations", {}).get("original_step6b", {}))
    density_lowcoh_spearman = get_nested(density_attr_corr, ["low_coherence_score", "spearman"])
    density_ant_spearman = get_nested(density_attr_corr, ["AntTrack", "spearman"])
    density_curvmax_spearman = get_nested(density_attr_corr, ["CurvatureMax", "spearman"])

    largest_lowcoh = (lowcoh.get("largest_components") or [{}])[0]
    lowcoh_voxel_count = float(lowcoh.get("lowcoh_voxel_count", 0.0) or 0.0)
    largest_lowcoh_fraction = (
        float(largest_lowcoh.get("voxel_count", 0.0) or 0.0) / lowcoh_voxel_count if lowcoh_voxel_count > 0 else None
    )

    validation = model_validation_summary(step6_summary)
    min_validation_r2 = min(
        [float(item["validation_r2"]) for item in validation.values() if isinstance(item.get("validation_r2"), int | float)],
        default=None,
    )

    input_alignment_pass = input_qc.get("status") == "pass"
    well_direction_ok = (
        real_density_coh is not None
        and real_density_ant is not None
        and real_density_coh < 0.0
        and real_density_ant > 0.0
    )
    random_validation_ok = min_validation_r2 is not None and min_validation_r2 >= 0.5
    original_density_lowcoh_related = original_top1_lowcoh is not None and original_top1_lowcoh >= 1.5
    original_density_ant_weak = original_top1_ant is None or original_top1_ant < 1.2
    original_density_curvature_weak = (
        original_top1_curvmax is None
        or original_top1_curvpos is None
        or max(original_top1_curvmax, original_top1_curvpos) < 1.0
    )
    lowcoh_has_giant_component = largest_lowcoh_fraction is not None and largest_lowcoh_fraction > 0.5

    step6a_position = (
        "usable_as_small_scale_background_with_caution"
        if input_alignment_pass and well_direction_ok and random_validation_ok
        else "not_locked_requires_model_qc_or_retraining"
    )
    step6a_retrain_required_now = not (input_alignment_pass and well_direction_ok and random_validation_ok)
    step6b_independent_required = original_density_ant_weak or original_density_curvature_weak
    step6c_independent_required = True

    decisions = {
        "step6a_position": step6a_position,
        "step6a_retrain_required_now": step6a_retrain_required_now,
        "step6a_use_original_density_as": "small_background_only" if not step6a_retrain_required_now else "not_locked",
        "step6b_independent_medium_prior_required": bool(step6b_independent_required),
        "step6b_primary_evidence": "AntTrack high value, supported by local low-coherence and curvature",
        "step6c_independent_large_prior_required": bool(step6c_independent_required),
        "step6c_primary_evidence": "original fault panels as hard constraints plus steep/non-horizontal low-coherence candidates",
        "step6d_required_contract": "preserve separate small/medium/large channels and scale labels; integrated density is compatibility only",
        "leave_one_well_validation": {
            "status": "not_executed_in_this_lightweight_qc",
            "reason": "true leave-one-well validation requires retraining grouped models and should be a separate expensive run before final model claims",
            "current_available_validation": "random row validation from existing Step6 summary only",
        },
    }

    recommendations: list[str] = []
    if step6a_position == "usable_as_small_scale_background_with_caution":
        recommendations.append("Freeze original Step6 density as Step6A small-scale background for the next Step7A pass.")
    else:
        recommendations.append("Pause Step7 and retrain or reweight Step6A before using it as background density.")
    if step6b_independent_required:
        recommendations.append("Do not let original Step6 high density control medium-scale fracture corridors; build Step6B from AntTrack-led seismic evidence.")
    if step6c_independent_required:
        recommendations.append("Keep Step6C independent and hard-constrained by original fault interpretation; do not infer large faults from Step6A density.")
    if lowcoh_has_giant_component:
        recommendations.append("Low-coherence has a giant connected component; apply geometry filtering before using low coherence for medium/large priors.")
    recommendations.append("Run grouped/leave-one-well validation before claiming Step6A model generalization in external reporting.")

    return {
        "status": "pass",
        "output_dir": str(output_dir),
        "inputs": {
            "config": str(DEFAULT_CONFIG),
            "input_alignment_qc": str(input_qc_path),
            "density_attribute_qc": str(density_qc_path),
            "well_correlation_qc": str(well_corr_path),
            "overlap_qc": str(overlap_path),
            "lowcoh_geometry_qc": str(lowcoh_path),
            "step6_summary": str(step6_summary_path),
        },
        "evidence": {
            "input_alignment_status": input_qc.get("status"),
            "training_scan_summary": scan,
            "virtual_to_real_row_ratio_seen": virtual_to_real_row_ratio,
            "target_block_source_counts": target_counts,
            "target_block_nominal_weight_sum": target_weights,
            "target_virtual_to_real_row_ratio": target_virtual_rows / target_real_rows if target_real_rows > 0 else None,
            "target_virtual_to_real_weight_ratio": target_virtual_weight / target_real_weight if target_real_weight > 0 else None,
            "random_validation_by_layer": validation,
            "min_random_validation_r2": min_validation_r2,
            "well_label_attribute_correlations": {
                "real_density_vs_coherence_pearson": real_density_coh,
                "real_density_vs_anttrack_pearson": real_density_ant,
                "real_flag_vs_coherence_pearson": real_flag_coh,
                "real_flag_vs_anttrack_pearson": real_flag_ant,
                "all_density_vs_coherence_pearson": all_density_coh,
                "all_density_vs_anttrack_pearson": all_density_ant,
            },
            "density_attribute_correlations_original_step6b": {
                "lowcoherence_spearman": density_lowcoh_spearman,
                "anttrack_spearman": density_ant_spearman,
                "curvaturemax_spearman": density_curvmax_spearman,
            },
            "top1_density_overlap_enrichment_original_step6b": {
                "coherence_le_q10": original_top1_lowcoh,
                "anttrack_ge_q90": original_top1_ant,
                "curvaturemax_ge_q90": original_top1_curvmax,
                "curvaturepos_ge_q90": original_top1_curvpos,
            },
            "top1_density_overlap_enrichment_lowcoh_steep": {
                "coherence_le_q10": lowcoh_steep_top1_lowcoh,
            },
            "lowcoh_component_geometry": {
                "component_count": lowcoh.get("component_count"),
                "lowcoh_voxel_count": lowcoh.get("lowcoh_voxel_count"),
                "horizontal_like_component_count": lowcoh.get("horizontal_like_component_count"),
                "steep_like_component_count": lowcoh.get("steep_like_component_count"),
                "largest_component": largest_lowcoh,
                "largest_component_fraction": largest_lowcoh_fraction,
            },
        },
        "decisions": decisions,
        "recommendations": recommendations,
        "checks": {
            "input_alignment_pass": input_alignment_pass,
            "random_validation_ok": random_validation_ok,
            "well_direction_ok": well_direction_ok,
            "original_density_lowcoh_related": original_density_lowcoh_related,
            "original_density_ant_weak": original_density_ant_weak,
            "original_density_curvature_weak": original_density_curvature_weak,
            "lowcoh_has_giant_component": lowcoh_has_giant_component,
        },
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, decision: dict[str, Any]) -> None:
    evidence = decision["evidence"]
    decisions = decision["decisions"]
    checks = decision["checks"]
    lines = [
        "# Step6 定位 QC 报告",
        "",
        "## 结论",
        "",
        f"- Step6A 定位：`{decisions['step6a_position']}`。",
        f"- Step6A 是否需要立即重训：`{decisions['step6a_retrain_required_now']}`。",
        f"- Step6B 是否必须独立构造：`{decisions['step6b_independent_medium_prior_required']}`。",
        f"- Step6C 是否必须独立构造：`{decisions['step6c_independent_large_prior_required']}`。",
        f"- Step6D 合同：{decisions['step6d_required_contract']}。",
        "",
        "## 关键证据",
        "",
        f"- 输入对齐状态：`{evidence['input_alignment_status']}`。",
        f"- 全样本虚拟井/真实井行数比：`{fmt(evidence['virtual_to_real_row_ratio_seen'])}`。",
        f"- demo 区虚拟井/真实井行数比：`{fmt(evidence['target_virtual_to_real_row_ratio'])}`。",
        f"- demo 区虚拟井/真实井名义权重比：`{fmt(evidence['target_virtual_to_real_weight_ratio'])}`。",
        f"- 最低随机验证 R2：`{fmt(evidence['min_random_validation_r2'])}`。",
        f"- 真实井 DensityLabel-Coherence Pearson：`{fmt(evidence['well_label_attribute_correlations']['real_density_vs_coherence_pearson'])}`。",
        f"- 真实井 DensityLabel-AntTrack Pearson：`{fmt(evidence['well_label_attribute_correlations']['real_density_vs_anttrack_pearson'])}`。",
        f"- 原始 Step6 最高 1% 密度与低相干 q10 富集：`{fmt(evidence['top1_density_overlap_enrichment_original_step6b']['coherence_le_q10'])}`。",
        f"- 原始 Step6 最高 1% 密度与蚂蚁体 q90 富集：`{fmt(evidence['top1_density_overlap_enrichment_original_step6b']['anttrack_ge_q90'])}`。",
        f"- 原始 Step6 最高 1% 密度与 CurvatureMax q90 富集：`{fmt(evidence['top1_density_overlap_enrichment_original_step6b']['curvaturemax_ge_q90'])}`。",
        f"- 低相干最大连通体占低相干体素比例：`{fmt(evidence['lowcoh_component_geometry']['largest_component_fraction'])}`。",
        "",
        "## 判断",
        "",
        f"- 输入对齐通过：`{checks['input_alignment_pass']}`。",
        f"- 随机验证达标：`{checks['random_validation_ok']}`。",
        f"- 井点方向一致：`{checks['well_direction_ok']}`。",
        f"- 原始密度与低相干有关系：`{checks['original_density_lowcoh_related']}`。",
        f"- 原始密度与蚂蚁体关系偏弱：`{checks['original_density_ant_weak']}`。",
        f"- 原始密度与曲率关系偏弱：`{checks['original_density_curvature_weak']}`。",
        f"- 低相干存在巨大连通体：`{checks['lowcoh_has_giant_component']}`。",
        "",
        "## 建议",
        "",
    ]
    lines.extend(f"- {item}" for item in decision["recommendations"])
    lines.extend(
        [
            "",
            "## 尚未完成的验证",
            "",
            "- 本次轻量 QC 没有执行真正的按井留一重训验证；当前只有原 Step6 summary 中的随机行验证。",
            "- 如果要对外证明模型泛化能力，需要单独跑 grouped/leave-one-well validation。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    config = read_json(args.config.resolve())

    input_qc_dir = output_dir / "input_alignment_qc"
    density_qc_dir = output_dir / "density_attribute_qc"
    ensure_dir(input_qc_dir)
    ensure_dir(density_qc_dir)

    if not args.skip_sub_qc:
        run_command(
            [
                sys.executable,
                str(CURRENT_DIR / "qc_multiscale_rebalance_inputs.py"),
                "--base-config",
                str(args.config.resolve()),
                "--multiscale-config",
                str(args.multiscale_config.resolve()),
                "--output-dir",
                str(input_qc_dir),
            ],
            CURRENT_DIR.parent.parent.parent,
        )
        run_command(
            [
                sys.executable,
                str(CURRENT_DIR / "qc_multiscale_density_vs_attributes.py"),
                "--config",
                str(args.config.resolve()),
                "--output-dir",
                str(density_qc_dir),
                "--max-well-correlation-rows",
                str(args.max_well_correlation_rows),
                "--correlation-sample-size",
                str(args.correlation_sample_size),
            ],
            CURRENT_DIR.parent.parent.parent,
        )

    decision = build_decision(output_dir, config)
    decision_path = output_dir / "step6_positioning_decision.json"
    report_path = output_dir / "step6_positioning_report.md"
    write_json(decision_path, decision)
    write_report(report_path, decision)
    print(f"[step6-positioning] decision={decision_path}", flush=True)
    print(f"[step6-positioning] report={report_path}", flush=True)
    print(f"[step6-positioning] step6a_position={decision['decisions']['step6a_position']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
