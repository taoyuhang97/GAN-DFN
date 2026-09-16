#!/usr/bin/env python3
"""P0-4（r2）与当前正式链的逐项对比报告。

把差异分成三类：

* `expected_diff`   —— 预期差异：新增审计文件、Step5/SourceSampleID 内容键、Step8 井控片位移；
* `unexpected_diff` —— 非预期差异：需要停下来查清；
* `same`            —— 完全一致。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAIGU = ROOT / "太古界"


def read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, **kwargs)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_diff(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> dict:
    out: dict[str, dict] = {}
    if len(left) != len(right):
        out["__row_count__"] = {"old": int(len(left)), "new": int(len(right)), "changed": int(len(right) - len(left))}
    for column in columns:
        if column not in left.columns or column not in right.columns:
            continue
        a = pd.to_numeric(left[column], errors="coerce")
        b = pd.to_numeric(right[column], errors="coerce")
        if len(a) != len(b):
            continue
        diff = (a - b).abs()
        out[column] = {
            "max_abs_diff": float(np.nanmax(diff)) if diff.notna().any() else 0.0,
            "changed_rows": int((diff.fillna(0) > 1e-9).sum()),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare P0-4 r2 outputs against the current formal chain.")
    parser.add_argument("--report", type=Path, default=TAIGU / "output_p04_r2_logs/p0_4_r2_report.json")
    args = parser.parse_args()

    sections: dict[str, dict] = {}
    unexpected: list[str] = []

    # ---------- Step5 ----------
    s5_old_dir = TAIGU / "step5_virtual_wells/output/taigu_step5_attribute_v3_common_contract"
    s5_new_dir = TAIGU / "step5_virtual_wells/output/taigu_step5_attribute_v3_common_contract_r2"
    old_summary = read_json(s5_old_dir / "taigu_step5_acceptance_summary.json") or {}
    new_summary = read_json(s5_new_dir / "taigu_step5_acceptance_summary.json") or {}
    step5 = {
        "unified_sample_count": {
            "old": old_summary.get("unified_sample_count"),
            "new": new_summary.get("unified_sample_count"),
        },
        "positive_fraction": {
            "old": old_summary.get("positive_fraction"),
            "new": new_summary.get("positive_fraction"),
        },
        "counts_by_kind": {
            "old": old_summary.get("unified_counts_by_kind"),
            "new": new_summary.get("unified_counts_by_kind"),
        },
        "new_audit_files": sorted(
            p.name for p in s5_new_dir.glob("*join*audit*.csv")
        ) if s5_new_dir.exists() else [],
        "segment_join_qc": new_summary.get("strong_supervision_segment_join_qc"),
        "id_key_changed": True,
    }
    if step5["unified_sample_count"]["old"] != step5["unified_sample_count"]["new"]:
        unexpected.append("Step5 统一样本行数发生变化")
    if step5["counts_by_kind"]["old"] != step5["counts_by_kind"]["new"]:
        unexpected.append("Step5 SourceKind 分布发生变化")
    # 除 ID 列外的逐列一致性
    old_samples = read_csv(s5_old_dir / "taigu_step5_unified_samples.csv")
    new_samples = read_csv(s5_new_dir / "taigu_step5_unified_samples.csv")
    if old_samples is not None and new_samples is not None and len(old_samples) == len(new_samples):
        id_columns = {"SourceSampleID", "SampleID", "VirtualWellName", "TrackWellName"}
        compare_columns = [c for c in old_samples.columns if c in new_samples.columns and c not in id_columns]
        diffs = numeric_diff(old_samples[compare_columns], new_samples[compare_columns], compare_columns)
        step5["non_id_column_diffs"] = {
            key: value for key, value in diffs.items() if value.get("changed_rows", 0) > 0 or key == "__row_count__"
        }
        if step5["non_id_column_diffs"]:
            unexpected.append(f"Step5 除 ID 列外出现数值差异: {list(step5['non_id_column_diffs'])}")
    sections["step5"] = step5

    # ---------- Step6A ----------
    a_old = read_json(TAIGU / "step6a_density_volume/output/taigu_step6a_attribute_v3/models/step6_two_stage_training_summary.json") or {}
    a_new = read_json(TAIGU / "step6a_density_volume/output/taigu_step6a_attribute_v3_r2/models/step6_two_stage_training_summary.json") or {}
    volume_compare = read_json(TAIGU / "output_p04_r2_logs/step6a_volume_compare.json") or {}
    sections["step6a"] = {
        "old_metrics": a_old.get("layer_metrics"),
        "new_metrics": a_new.get("layer_metrics"),
        "volume_compare": volume_compare,
    }
    if volume_compare and not volume_compare.get("identical", False):
        # 体变了不算错，但必须触发 6B/6C/7B/7C 重跑（runner 已处理），这里记录
        sections["step6a"]["volume_changed"] = True

    # ---------- Step7A ----------
    p_old = read_json(TAIGU / "step7a_small_scale_dfn/output/taigu_step7a_attribute_v3_regen_full/step7a_summary.json") or {}
    p_new = read_json(TAIGU / "step7a_small_scale_dfn/output/taigu_step7a_attribute_v3_regen_full_r2/step7a_summary.json") or {}
    sections["step7a"] = {
        "old_patch_count": p_old.get("accepted_patch_count"),
        "new_patch_count": p_new.get("accepted_patch_count"),
        "segment_join_qc": p_new.get("segment_join_qc"),
    }
    if p_old.get("accepted_patch_count") != p_new.get("accepted_patch_count"):
        unexpected.append("Step7A 片数发生变化")

    # ---------- Step7D ----------
    f_old = read_json(TAIGU / "step7d_multiscale_fused_dfn/output/taigu_step7d_fused_v1/fused_multiscale_summary.json") or {}
    f_new = read_json(TAIGU / "step7d_multiscale_fused_dfn/output/taigu_step7d_fused_v1_r2/fused_multiscale_summary.json") or {}
    sections["step7d"] = {
        "old_patch_count": f_old.get("patch_count"),
        "new_patch_count": f_new.get("patch_count"),
        "old_fault_fingerprint": f_old.get("original_fault_geometry_fingerprint"),
        "new_fault_fingerprint": f_new.get("original_fault_geometry_fingerprint"),
    }
    if f_old.get("patch_count") != f_new.get("patch_count"):
        unexpected.append("Step7D 融合片数发生变化")

    # ---------- Step8 ----------
    w_old = read_json(TAIGU / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1/well_corrected_dfn_summary.json") or {}
    w_new = read_json(TAIGU / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1_r2/well_corrected_dfn_summary.json") or {}
    old_patches = read_csv(TAIGU / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1/well_corrected_dfn_fracture_patches.csv")
    new_patches = read_csv(TAIGU / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1_r2/well_corrected_dfn_fracture_patches.csv")
    step8: dict = {
        "old_control_count": (w_old.get("well_controls") or {}).get("control_point_count"),
        "new_control_count": (w_new.get("well_controls") or {}).get("control_point_count"),
        "old_action_counts": (w_old.get("correction_actions") or {}).get("action_counts"),
        "new_action_counts": (w_new.get("correction_actions") or {}).get("action_counts"),
        "new_md_join_qc": (w_new.get("well_controls") or {}).get("md_join_qc"),
        "new_status": w_new.get("status"),
    }
    if old_patches is not None and new_patches is not None:
        if len(old_patches) != len(new_patches):
            unexpected.append("Step8 片数发生变化")
        else:
            index = ["PatchID"]
            old_indexed = old_patches.set_index(index).sort_index()
            new_indexed = new_patches.set_index(index).sort_index()
            frame_diff = numeric_diff(old_indexed, new_indexed, ["CenterX", "CenterY", "CenterTime", "SourceDensity"])
            moved = int((pd.to_numeric(old_indexed["CenterX"], errors="coerce")
                         - pd.to_numeric(new_indexed["CenterX"], errors="coerce")).abs().fillna(0).gt(1e-9).sum())
            step8["moved_patch_count"] = moved
            step8["moved_patch_fraction"] = float(moved) / max(len(old_patches), 1)
            step8["geometry_diffs"] = frame_diff
            from_patch = TAIGU / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1"
            to_patch = TAIGU / "step8_well_correction/output/taigu_step8_attribute_multiscale_v1_r2"
            moved_ids = old_indexed.index[
                (pd.to_numeric(old_indexed["CenterX"], errors="coerce")
                 - pd.to_numeric(new_indexed["CenterX"], errors="coerce")).abs().fillna(0).gt(1e-9)
            ]
            moved_frame = new_indexed.loc[moved_ids].reset_index()[
                [c for c in ("PatchID", "CorrectionAction", "WellControlWellName", "LayerGroup",
                             "CenterX", "CenterY", "CenterTime") if c in new_indexed.reset_index().columns]
            ]
            moved_frame.to_csv(TAIGU / "output_p04_r2_logs/step8_moved_control_patches.csv",
                               index=False, encoding="utf-8-sig")
            step8["moved_patch_list_csv"] = str(TAIGU / "output_p04_r2_logs/step8_moved_control_patches.csv")
            medium_large_moved = int(
                (pd.to_numeric(old_indexed["CenterX"], errors="coerce")
                 - pd.to_numeric(new_indexed["CenterX"], errors="coerce")).abs().fillna(0).gt(1e-9)[
                    old_indexed["FractureScale"].astype(str).str.lower().ne("small")
                ].sum()
            ) if "FractureScale" in old_indexed.columns else -1
            step8["medium_large_moved_count"] = medium_large_moved
            if medium_large_moved > 0:
                unexpected.append("Step8 中/大尺度片发生位移（不允许）")
    sections["step8"] = step8

    # ---------- Step9 ----------
    s9_new = read_json(TAIGU / "step9_sections/output/taigu_step9_multibackground_v2_r2/section_summary.json") or {}
    sections["step9"] = {
        "status": s9_new.get("status"),
        "counts": s9_new.get("counts"),
        "imaging_point_join_qc": s9_new.get("imaging_point_join_qc"),
        "image_count": len(s9_new.get("images", []) or []),
    }
    if s9_new and s9_new.get("status") != "pass":
        unexpected.append("Step9 status 不是 pass")

    report = {
        "expected_diff": [
            "Step5/Step5B 的 SourceSampleID 改为内容键（井名+深度）",
            "Step8 井控片微偏移键改为内容键 -> 部分井控片位移（见 step8_moved_control_patches.csv）",
            "新增测井段回接审计文件（Step5/7A/8/9）",
        ],
        "unexpected_diff": unexpected,
        "sections": sections,
        "verdict": "pass" if not unexpected else "review_required",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "unexpected_diff": unexpected}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
