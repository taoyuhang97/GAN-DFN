#!/usr/bin/env python3
"""只读诊断：跨测次重叠区的拼接口径比较（不改动任何流程产物）。

对同时满足"有成像监督 + 有多个重叠测次"的井，在重叠深度区间上比较：

  ① 现状：Step4 合并表（跨测次等权平均）
  ② 按段独立：每个测次各自的预测曲线，以及样本级（每段一行）的秩相关
  ③ 择一：取与成像密度秩相关更高的那一段（作为上界参考）

参考量同时给出"非重叠区"的秩相关，用于判断重叠区是否真的被降级。
评价指标以 **Spearman 秩相关** 为主（预测密度与成像密度量纲不同，秩相关免疫尺度），
并附原始曲线在同深度上的跨测次差异，量化"批次效应"。

用法：
    python3 太古界/diagnostics/segment_overlap_diagnosis.py \
        --output 太古界/diagnostics/segment_overlap_20260916
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
TAIGU = ROOT / "太古界"
STEP2 = TAIGU / "step2_well_log_segments/output/taigu_step2_regular_v3"
STEP3_GROUPS = TAIGU / "step3_imaging_groups/output/taigu_step3_imaging_v3/groups"
STEP4_DETAIL = (
    TAIGU / "step4_fracture_prediction/output/taigu_step4_gr_rd_rs_v1/predictions/"
    "all_wells_source_detail_predictions.csv"
)
STEP4_MERGED = (
    TAIGU / "step4_fracture_prediction/output/taigu_step4_gr_rd_rs_v1/predictions/"
    "all_wells_merged_density_prediction.csv"
)

# 有成像监督且存在重叠测次的井（段数来自 Step2 段清单）
TARGET_WELLS = ["埕北古斜405", "埕北313", "埕北310"]
RAW_CURVES = ["GR", "RD", "RS"]


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, **kwargs)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20 or np.unique(x[mask]).size < 5 or np.unique(y[mask]).size < 5:
        return None
    value = spearmanr(x[mask], y[mask]).statistic
    return float(value) if np.isfinite(value) else None


def nearest_match(source_tvd: np.ndarray, source_val: np.ndarray, target_tvd: np.ndarray, tol: float):
    """把 source 曲线按最近深度匹配到 target 网格。"""
    order = np.argsort(source_tvd, kind="mergesort")
    grid = source_tvd[order]
    values = source_val[order]
    pos = np.searchsorted(grid, target_tvd)
    out = np.full(target_tvd.size, np.nan)
    for i, p in enumerate(pos):
        best = None
        for cand in (p - 1, p):
            if 0 <= cand < grid.size:
                dist = abs(grid[cand] - target_tvd[i])
                if dist <= tol and (best is None or dist < best[0]):
                    best = (dist, values[cand])
        if best is not None:
            out[i] = best[1]
    return out


def diagnose_well(well: str, detail: pd.DataFrame, merged: pd.DataFrame, tol: float) -> dict:
    group_files = sorted(STEP3_GROUPS.glob(f"{well}_*.csv"))
    report: dict = {"well": well, "layers": {}}
    for group_file in group_files:
        group = read_csv(group_file)
        if group.empty:
            continue
        # 同一口井可能有多个同名地层的 group（如 313 的 g001/g002 都是"上部复合层"），
        # 必须用 group 文件名做键，否则后一个会覆盖前一个。
        layer = group_file.stem
        layer_name = str(group["StrataName"].dropna().astype(str).mode().iloc[0])
        imaging = (
            group[["TVD", "Density", "GT_POINT_FLAG"]]
            .assign(TVD=lambda d: pd.to_numeric(d["TVD"], errors="coerce"),
                    Density=lambda d: pd.to_numeric(d["Density"], errors="coerce"))
            .dropna(subset=["TVD", "Density"])
            .groupby("TVD", as_index=False)
            .agg(Density=("Density", "mean"), GT=("GT_POINT_FLAG", "max"))
            .sort_values("TVD")
            .reset_index(drop=True)
        )
        well_detail = detail[(detail["WellName"].astype(str) == well)]
        well_merged = merged[merged["WellName"].astype(str) == well].copy()
        if well_detail.empty or imaging.empty:
            continue
        runs = sorted(well_detail["SegmentID"].astype(str).unique())
        # 该地层内的测次（用 TVD 范围判断）
        lo, hi = float(imaging["TVD"].min()), float(imaging["TVD"].max())
        run_in_range = {
            run: well_detail[well_detail["SegmentID"].astype(str).eq(run)]
            for run in runs
        }
        run_in_range = {
            run: frame for run, frame in run_in_range.items()
            if len(frame) and frame["TVD"].max() >= lo and frame["TVD"].min() <= hi
        }
        if len(run_in_range) < 2:
            report["layers"][layer] = {
                "layer_name": layer_name,
                "runs_in_layer": sorted(run_in_range),
                "note": "该地层只有一个测次，无重叠可比",
            }
            continue

        # 成像网格上匹配各测次预测与合并预测
        matched: dict[str, np.ndarray] = {}
        for run, frame in run_in_range.items():
            matched[run] = nearest_match(
                frame["TVD"].to_numpy(float), frame["PredDensity"].to_numpy(float),
                imaging["TVD"].to_numpy(float), tol,
            )
        matched["merged_avg"] = nearest_match(
            well_merged["TVD"].to_numpy(float), well_merged["PredDensity"].to_numpy(float),
            imaging["TVD"].to_numpy(float), max(tol, 0.3),
        )
        imaging_density = imaging["Density"].to_numpy(float)

        # 覆盖率：至少两个测次都有预测的深度 = 重叠区
        run_keys = sorted(run_in_range)
        run_matrix = np.vstack([matched[run] for run in run_keys])
        overlap_mask = np.isfinite(run_matrix).sum(axis=0) >= 2
        single_mask = np.isfinite(run_matrix).sum(axis=0) == 1
        merged_mask = np.isfinite(matched["merged_avg"])

        layer_report: dict = {
            "layer_name": layer_name,
            "runs": run_keys,
            "imaging_rows": int(len(imaging)),
            "tvd_range": [lo, hi],
            "overlap_depth_count": int(overlap_mask.sum()),
            "single_run_depth_count": int(single_mask.sum()),
            "overlap_spearman_vs_imaging": {},
            "single_run_spearman_vs_imaging": {},
            "run_pair_agreement": {},
            "raw_curve_batch_effect": {},
        }

        # ① 现状（合并平均）在重叠区 / 全区的一致性
        layer_report["overlap_spearman_vs_imaging"]["merged_avg"] = spearman(
            matched["merged_avg"][overlap_mask & merged_mask], imaging_density[overlap_mask & merged_mask]
        )
        layer_report["single_run_spearman_vs_imaging"]["merged_avg"] = spearman(
            matched["merged_avg"][single_mask & merged_mask], imaging_density[single_mask & merged_mask]
        )
        # ② 各测次独立
        for run in run_keys:
            mask_o = overlap_mask & np.isfinite(matched[run])
            mask_s = single_mask & np.isfinite(matched[run])
            layer_report["overlap_spearman_vs_imaging"][run] = spearman(
                matched[run][mask_o], imaging_density[mask_o]
            )
            if mask_s.sum() >= 20:
                layer_report["single_run_spearman_vs_imaging"][run] = spearman(
                    matched[run][mask_s], imaging_density[mask_s]
                )
        # 样本级（每段一行，等价"按段独立、不做深度平均"）
        sample_level = []
        for run in run_keys:
            frame = run_in_range[run]
            sample_level.append(
                pd.DataFrame({
                    "TVD": frame["TVD"].to_numpy(float),
                    "PredDensity": frame["PredDensity"].to_numpy(float),
                    "run": run,
                })
            )
        sample_frame = pd.concat(sample_level, ignore_index=True)
        sample_frame["ImagingDensity"] = nearest_match(
            imaging["TVD"].to_numpy(float), imaging_density,
            sample_frame["TVD"].to_numpy(float), tol,
        )
        sample_frame = sample_frame.dropna()
        layer_report["sample_level_spearman_vs_imaging"] = spearman(
            sample_frame["PredDensity"].to_numpy(float), sample_frame["ImagingDensity"].to_numpy(float)
        )
        layer_report["sample_level_rows"] = int(len(sample_frame))

        # ③ 择一：取重叠区秩相关更高的段（上界参考）
        overlap_corr = {
            run: layer_report["overlap_spearman_vs_imaging"].get(run) for run in run_keys
        }
        valid = {k: v for k, v in overlap_corr.items() if v is not None}
        layer_report["best_single_run"] = max(valid, key=valid.get) if valid else None
        layer_report["best_single_run_spearman"] = max(valid.values()) if valid else None

        # 测次两两一致性与原始曲线批次效应（重叠深度）
        for i in range(len(run_keys)):
            for j in range(i + 1, len(run_keys)):
                a, b = run_keys[i], run_keys[j]
                mask = np.isfinite(matched[a]) & np.isfinite(matched[b])
                if mask.sum() < 20:
                    continue
                pair = f"{a}__vs__{b}"
                layer_report["run_pair_agreement"][pair] = {
                    "overlap_depth_count": int(mask.sum()),
                    "predicted_density_spearman": spearman(matched[a][mask], matched[b][mask]),
                }
                fa, fb = run_in_range[a], run_in_range[b]
                for curve in RAW_CURVES:
                    if curve not in fa.columns or curve not in fb.columns:
                        continue
                    paired_b = nearest_match(
                        fb["TVD"].to_numpy(float), fb[curve].to_numpy(float),
                        fa["TVD"].to_numpy(float), tol,
                    )
                    mask2 = np.isfinite(paired_b) & np.isfinite(fa[curve].to_numpy(float))
                    if mask2.sum() < 20:
                        continue
                    diff = np.abs(fa[curve].to_numpy(float)[mask2] - paired_b[mask2])
                    layer_report["raw_curve_batch_effect"][f"{pair}::{curve}"] = {
                        "matched_rows": int(mask2.sum()),
                        "median_abs_diff": float(np.median(diff)),
                        "median_relative_diff": float(
                            np.median(diff / (np.abs(fa[curve].to_numpy(float)[mask2]) + 1e-9))
                        ),
                        "spearman": spearman(
                            fa[curve].to_numpy(float)[mask2], paired_b[mask2]
                        ),
                    }
        report["layers"][layer] = layer_report
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=TAIGU / "diagnostics/segment_overlap_20260916")
    # 默认容差取"半采样步长"：316/405 的段间 TVD 网格偏移约 0.014 m，
    # 310 的两段偏移 0.042 m，故 0.06 m 可覆盖且不会跨到下一个真实采样点。
    parser.add_argument("--tolerance-m", type=float, default=0.06)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    detail = read_csv(STEP4_DETAIL)
    merged = read_csv(STEP4_MERGED)
    detail["SegmentID"] = detail["SegmentID"].astype(str)
    dup = int(detail.duplicated(["WellName", "SegmentID", "TVD"]).sum())
    print(f"Step4 逐段明细 {len(detail)} 行；重复 (井,段,TVD) {dup} 行")

    report = {
        "inputs": {
            "step4_detail": str(STEP4_DETAIL),
            "step4_merged": str(STEP4_MERGED),
            "step3_groups": str(STEP3_GROUPS),
            "tolerance_m": float(args.tolerance_m),
        },
        "caveat": (
            "这些成像井同时参与了 Step4 训练，所以与成像密度的秩相关是样本内的乐观值；"
            "本诊断只用于比较同一批深度上不同拼接口径的相对优劣。"
        ),
        "wells": {},
    }
    for well in TARGET_WELLS:
        report["wells"][well] = diagnose_well(well, detail, merged, float(args.tolerance_m))

    out_json = args.output / "segment_overlap_diagnosis.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # 控制台摘要
    for well, payload in report["wells"].items():
        print(f"\n=== {well} ===")
        for layer, info in payload["layers"].items():
            if "overlap_depth_count" not in info:
                print(f"  [{layer}] {info.get('note')} runs={info.get('runs_in_layer')}")
                continue
            print(f"  [{layer}] 重叠深度 {info['overlap_depth_count']} / 单段深度 {info['single_run_depth_count']}"
                  f"  runs={info['runs']}")
            print("    重叠区 Spearman(预测, 成像): "
                  + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}=n/a"
                              for k, v in info["overlap_spearman_vs_imaging"].items()))
            print("    单段区 Spearman(预测, 成像): "
                  + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}=n/a"
                              for k, v in info["single_run_spearman_vs_imaging"].items()))
            print(f"    样本级（每段一行）Spearman={info.get('sample_level_spearman_vs_imaging')} "
                  f"rows={info.get('sample_level_rows')}")
            print(f"    最优单段={info.get('best_single_run')} (ρ={info.get('best_single_run_spearman')})")
            for pair, pair_info in info["run_pair_agreement"].items():
                rho = pair_info["predicted_density_spearman"]
                print(f"    测次间一致 {pair}: n={pair_info['overlap_depth_count']} ρ={rho}")
            for key, curve_info in list(info["raw_curve_batch_effect"].items())[:6]:
                print(f"    原始曲线批次效应 {key}: 中位绝对差={curve_info['median_abs_diff']:.4g} "
                      f"相对差={curve_info['median_relative_diff']:.1%} ρ={curve_info['spectrum'] if False else curve_info['spearman']}")
    print(f"\n报告已写入 {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
