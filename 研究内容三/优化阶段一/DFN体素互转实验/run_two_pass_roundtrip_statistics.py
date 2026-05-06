# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

from roundtrip_common import (
    DEFAULT_OUTPUT_ROOT as LEGACY_DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRACE_HEADER_CSV,
    GridSpec,
    build_grid_spec,
    build_voxel_overlap_metrics,
    compute_time_bounds,
    compute_unit_bounds_from_patch_table,
    compute_unit_bounds_from_trace_header,
    fit_voxel_components_to_patches,
    load_layer_table,
    load_patch_table,
    load_unit_summary,
    patch_angle_diff_deg,
    rasterize_patches_to_voxel,
    resolve_layer_surface_pair_key_from_row,
    summarize_patch_statistics,
    write_csv_utf8,
    write_json,
)


DEFAULT_ANALYSIS_UNIT_DFN_ROOT = Path(
    "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/单元DFN构建/批量生成_新层位重拆分"
)
DEFAULT_OUTPUT_ROOT = LEGACY_DEFAULT_OUTPUT_ROOT / "双轮统计"
DEFAULT_CHANNELS = "occupancy"
DEFAULT_MATCH_COST_THRESHOLD = 2.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="筛选代表性单元，并执行 DFN->体素->DFN 两轮互转统计。")
    parser.add_argument("--unit-dfn-root", type=Path, default=DEFAULT_ANALYSIS_UNIT_DFN_ROOT)
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="two_pass_roundtrip")
    parser.add_argument("--unit-ids", nargs="*", default=[])
    parser.add_argument("--xy-resolution", type=int, default=24)
    parser.add_argument("--z-step-ms", type=float, default=0.2)
    parser.add_argument("--z-padding-ms", type=float, default=0.0)
    parser.add_argument("--thickness-vox", type=float, default=1.0)
    parser.add_argument("--channels", choices=["occupancy", "occupancy_normals"], default=DEFAULT_CHANNELS)
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--min-component-voxels", type=int, default=6)
    parser.add_argument("--connectivity", type=int, default=1)
    parser.add_argument("--max-scan-count", type=int, default=6)
    return parser


def build_unit_screening(unit_dfn_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit_dir in sorted([path for path in Path(unit_dfn_root).iterdir() if path.is_dir()]):
        patch_csv = unit_dir / "unit_dfn_patches.csv"
        layer_csv = unit_dir / "unit_layers_input.csv"
        summary_json = unit_dir / "unit_dfn_summary.json"
        files_complete = patch_csv.exists() and layer_csv.exists() and summary_json.exists()
        if not files_complete:
            rows.append(
                {
                    "单元ID": unit_dir.name,
                    "文件完整": False,
                    "裂缝片数": 0,
                    "层段数": 0,
                    "数据模式": "",
                    "可靠性类别": "",
                    "是否适合统计": False,
                    "推荐分组": "",
                    "推荐说明": "缺少 unit_dfn_patches / unit_layers_input / unit_dfn_summary 文件",
                }
            )
            continue

        patch_df = load_patch_table(patch_csv)
        layers_df = load_layer_table(unit_dir)
        summary = load_unit_summary(unit_dir)
        interval_count = 0
        if "GeoIntervalKey" in patch_df.columns:
            interval_series = patch_df["GeoIntervalKey"].fillna("").astype(str).str.strip()
            interval_count = int(interval_series.mask(interval_series.eq(""), pd.NA).dropna().nunique())
        rows.append(
            {
                "单元ID": unit_dir.name,
                "文件完整": True,
                "裂缝片数": int(len(patch_df)),
                "层段数": int(interval_count if interval_count > 0 else len(layers_df)),
                "数据模式": str(summary.get("DataMode", "")),
                "可靠性类别": str(summary.get("ReliabilityClass", "")),
            }
        )

    screening_df = pd.DataFrame(rows)
    if screening_df.empty:
        return screening_df

    screening_df["是否适合统计"] = (
        screening_df["文件完整"].fillna(False)
        & screening_df["裂缝片数"].fillna(0).ge(150)
        & screening_df["层段数"].fillna(0).ge(5)
        & screening_df["数据模式"].isin(["real_virtual", "virtual_only"])
        & screening_df["可靠性类别"].fillna("").astype(str).str.contains("controlled")
    )
    screening_df["推荐分组"] = ""
    screening_df["推荐说明"] = ""
    return screening_df


def pick_representative_units(screening_df: pd.DataFrame) -> pd.DataFrame:
    if screening_df.empty:
        return pd.DataFrame(
            columns=[
                "单元ID",
                "裂缝片数",
                "层段数",
                "数据模式",
                "可靠性类别",
                "推荐分组",
                "推荐说明",
            ]
        )

    eligible = screening_df[screening_df["是否适合统计"]].copy()
    if eligible.empty:
        return pd.DataFrame(columns=list(screening_df.columns))

    picked_rows: list[pd.Series] = []
    quantile_targets = [
        ("低复杂度", 0.15),
        ("中等复杂度", 0.50),
        ("高复杂度", 0.85),
    ]
    for data_mode in ["real_virtual", "virtual_only"]:
        subset = eligible[eligible["数据模式"] == data_mode].copy()
        if subset.empty:
            continue
        used_ids: set[str] = set()
        for level_name, quantile_value in quantile_targets:
            target_value = float(subset["裂缝片数"].quantile(quantile_value))
            candidates = subset[~subset["单元ID"].isin(used_ids)].copy()
            if candidates.empty:
                continue
            candidates["目标距离"] = (candidates["裂缝片数"].astype(float) - target_value).abs()
            selected = candidates.sort_values(["目标距离", "层段数", "裂缝片数"], ascending=[True, False, True]).iloc[0]
            used_ids.add(str(selected["单元ID"]))
            picked_rows.append(selected.copy())

    if not picked_rows:
        return pd.DataFrame(columns=list(screening_df.columns))

    recommended_df = pd.DataFrame(picked_rows).copy()
    recommended_df["推荐分组"] = recommended_df.apply(
        lambda row: ("实井约束" if row["数据模式"] == "real_virtual" else "虚拟井主导")
        + "_"
        + infer_complexity_group_name(recommended_df, row["单元ID"]),
        axis=1,
    )
    recommended_df["推荐说明"] = recommended_df.apply(
        lambda row: (
            f"裂缝片数={int(row['裂缝片数'])}，层段数={int(row['层段数'])}，"
            f"属于 {('实井约束' if row['数据模式'] == 'real_virtual' else '虚拟井主导')} 单元，"
            "适合用于两轮互转的代表性统计。"
        ),
        axis=1,
    )
    return recommended_df[
        ["单元ID", "裂缝片数", "层段数", "数据模式", "可靠性类别", "推荐分组", "推荐说明"]
    ].reset_index(drop=True)


def infer_complexity_group_name(recommended_df: pd.DataFrame, unit_id: str) -> str:
    patch_value = float(recommended_df.loc[recommended_df["单元ID"] == unit_id, "裂缝片数"].iloc[0])
    mode = str(recommended_df.loc[recommended_df["单元ID"] == unit_id, "数据模式"].iloc[0])
    subset = recommended_df[recommended_df["数据模式"] == mode].sort_values("裂缝片数").reset_index(drop=True)
    order = subset["单元ID"].tolist()
    if unit_id not in order:
        return "未知复杂度"
    idx = order.index(unit_id)
    if idx == 0:
        return "低复杂度"
    if idx == len(order) - 1:
        return "高复杂度"
    _ = patch_value
    return "中等复杂度"


def build_grid_for_unit(
    patch_df: pd.DataFrame,
    unit_id: str,
    unit_dir: Path,
    trace_header_csv: Path,
    xy_resolution: int,
    z_step_ms: float,
    z_padding_ms: float,
) -> tuple[GridSpec, pd.DataFrame, dict[str, Any]]:
    block_x = int(patch_df["BlockX"].iloc[0])
    block_y = int(patch_df["BlockY"].iloc[0])
    layers_df = load_layer_table(unit_dir)
    unit_summary = load_unit_summary(unit_dir)
    try:
        x_min, x_max, y_min, y_max = compute_unit_bounds_from_trace_header(
            trace_header_csv,
            block_x=block_x,
            block_y=block_y,
        )
    except Exception:
        x_min, x_max, y_min, y_max = compute_unit_bounds_from_patch_table(patch_df)

    z_min, z_max = compute_time_bounds(patch_df, layers_df, z_padding_ms=z_padding_ms)
    grid = build_grid_spec(
        unit_id=unit_id,
        block_x=block_x,
        block_y=block_y,
        x_bounds=(x_min, x_max),
        y_bounds=(y_min, y_max),
        z_bounds=(z_min, z_max),
        xy_resolution=xy_resolution,
        z_step_ms=z_step_ms,
    )
    return grid, layers_df, unit_summary


def build_patch_match_detail(
    input_patches: pd.DataFrame,
    output_patches: pd.DataFrame,
    grid: GridSpec,
    max_cost_threshold: float = DEFAULT_MATCH_COST_THRESHOLD,
) -> dict[str, Any]:
    if input_patches.empty or output_patches.empty:
        return {
            "matched_patch_count": 0,
            "same_layer_match_count": 0,
            "center_offsets": [],
            "azimuth_diffs": [],
            "dip_diffs": [],
            "accepted_pairs": [],
        }

    span_x = max(grid.x_max - grid.x_min, 1e-6)
    span_y = max(grid.y_max - grid.y_min, 1e-6)
    span_t = max(grid.z_max - grid.z_min, 1e-6)
    cost = np.zeros((len(input_patches), len(output_patches)), dtype=float)
    for src_idx, (_, src) in enumerate(input_patches.iterrows()):
        src_layer_key = resolve_layer_surface_pair_key_from_row(src)
        for dst_idx, (_, dst) in enumerate(output_patches.iterrows()):
            dx = (float(src["CenterX"]) - float(dst["CenterX"])) / span_x
            dy = (float(src["CenterY"]) - float(dst["CenterY"])) / span_y
            dt = (float(src["CenterTIME"]) - float(dst["CenterTIME"])) / span_t
            center_cost = math.sqrt(dx * dx + dy * dy + dt * dt)
            az_cost = patch_angle_diff_deg(float(src["Azimuth"]), float(dst["Azimuth"])) / 180.0
            dip_cost = abs(float(src["Dip"]) - float(dst["Dip"])) / 90.0
            interval_penalty = 0.0
            if src_layer_key != resolve_layer_surface_pair_key_from_row(dst):
                interval_penalty = 0.5
            cost[src_idx, dst_idx] = center_cost + 0.25 * az_cost + 0.25 * dip_cost + interval_penalty

    if linear_sum_assignment is not None:
        row_idx, col_idx = linear_sum_assignment(cost)
    else:  # pragma: no cover
        row_idx = np.arange(min(cost.shape[0], cost.shape[1]))
        col_idx = np.argmin(cost[row_idx], axis=1)

    center_offsets: list[float] = []
    azimuth_diffs: list[float] = []
    dip_diffs: list[float] = []
    accepted_pairs: list[dict[str, Any]] = []
    same_layer_match_count = 0
    for src_idx, dst_idx in zip(row_idx, col_idx):
        pair_cost = float(cost[int(src_idx), int(dst_idx)])
        if pair_cost > float(max_cost_threshold):
            continue
        src = input_patches.iloc[int(src_idx)]
        dst = output_patches.iloc[int(dst_idx)]
        dx = float(dst["CenterX"]) - float(src["CenterX"])
        dy = float(dst["CenterY"]) - float(src["CenterY"])
        dt = float(dst["CenterTIME"]) - float(src["CenterTIME"])
        center_offset = math.sqrt(dx * dx + dy * dy + dt * dt)
        azimuth_diff = patch_angle_diff_deg(float(src["Azimuth"]), float(dst["Azimuth"]))
        dip_diff = abs(float(src["Dip"]) - float(dst["Dip"]))
        same_layer = resolve_layer_surface_pair_key_from_row(src) == resolve_layer_surface_pair_key_from_row(dst)
        if same_layer:
            same_layer_match_count += 1
        center_offsets.append(center_offset)
        azimuth_diffs.append(azimuth_diff)
        dip_diffs.append(dip_diff)
        accepted_pairs.append(
            {
                "source_patch_id": str(src.get("PatchID", "")),
                "target_patch_id": str(dst.get("PatchID", "")),
                "cost": pair_cost,
                "center_offset": center_offset,
                "azimuth_diff": azimuth_diff,
                "dip_diff": dip_diff,
                "same_layer": bool(same_layer),
            }
        )

    return {
        "matched_patch_count": int(len(accepted_pairs)),
        "same_layer_match_count": int(same_layer_match_count),
        "center_offsets": center_offsets,
        "azimuth_diffs": azimuth_diffs,
        "dip_diffs": dip_diffs,
        "accepted_pairs": accepted_pairs,
    }


def build_chinese_metric_summary(
    before_patches: pd.DataFrame,
    after_patches: pd.DataFrame,
    before_occupancy: np.ndarray,
    after_occupancy: np.ndarray,
    grid: GridSpec,
    occupancy_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    voxel_metrics = build_voxel_overlap_metrics(
        input_occupancy=before_occupancy,
        output_occupancy=after_occupancy,
        threshold=occupancy_threshold,
    )
    match_detail = build_patch_match_detail(
        input_patches=before_patches,
        output_patches=after_patches,
        grid=grid,
        max_cost_threshold=DEFAULT_MATCH_COST_THRESHOLD,
    )
    before_patch_count = int(len(before_patches))
    matched_patch_count = int(match_detail["matched_patch_count"])
    same_layer_match_count = int(match_detail["same_layer_match_count"])

    def safe_median(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None

    chinese_summary = {
        "转化前裂缝片数": before_patch_count,
        "转化后裂缝片数": int(len(after_patches)),
        "裂缝体素综合重合度": voxel_metrics.get("voxel_iou"),
        "原始裂缝保留率": voxel_metrics.get("voxel_recall"),
        "回转裂缝有效命中率": voxel_metrics.get("voxel_precision"),
        "裂缝片对应率": float(matched_patch_count / before_patch_count) if before_patch_count else None,
        "裂缝中心偏移中位数": safe_median(match_detail["center_offsets"]),
        "裂缝倾向偏差中位数": safe_median(match_detail["azimuth_diffs"]),
        "裂缝倾角偏差中位数": safe_median(match_detail["dip_diffs"]),
        "层位归属一致率": float(same_layer_match_count / matched_patch_count) if matched_patch_count else None,
    }
    return chinese_summary, match_detail["accepted_pairs"]


def is_second_pass_stable(metric_row: dict[str, Any], tol: float = 1e-6) -> bool:
    must_be_one = [
        metric_row.get("裂缝体素综合重合度"),
        metric_row.get("原始裂缝保留率"),
        metric_row.get("回转裂缝有效命中率"),
        metric_row.get("裂缝片对应率"),
        metric_row.get("层位归属一致率"),
    ]
    must_be_zero = [
        metric_row.get("裂缝中心偏移中位数"),
        metric_row.get("裂缝倾向偏差中位数"),
        metric_row.get("裂缝倾角偏差中位数"),
    ]
    for value in must_be_one:
        if value is None or not np.isfinite(float(value)) or abs(float(value) - 1.0) > tol:
            return False
    for value in must_be_zero:
        if value is None or not np.isfinite(float(value)) or abs(float(value)) > tol:
            return False
    return True


def execute_two_pass_roundtrip_for_unit(
    unit_id: str,
    unit_dir: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    patch_csv = unit_dir / "unit_dfn_patches.csv"
    input_patch_df = load_patch_table(patch_csv)
    if input_patch_df.empty:
        raise ValueError(f"裂缝片文件为空: {patch_csv}")

    grid, layers_df, unit_summary = build_grid_for_unit(
        patch_df=input_patch_df,
        unit_id=unit_id,
        unit_dir=unit_dir,
        trace_header_csv=args.trace_header_csv,
        xy_resolution=args.xy_resolution,
        z_step_ms=args.z_step_ms,
        z_padding_ms=args.z_padding_ms,
    )
    include_normals = args.channels == "occupancy_normals"

    write_csv_utf8(input_patch_df, output_dir / "round0_input_patches.csv")

    ref_stats_0 = summarize_patch_statistics(input_patch_df)
    occ_0, normals_0, _ = rasterize_patches_to_voxel(
        patch_df=input_patch_df,
        grid=grid,
        thickness_vox=args.thickness_vox,
        include_normals=include_normals,
    )
    round1_patch_df, _ = fit_voxel_components_to_patches(
        occupancy=occ_0,
        grid=grid,
        threshold=args.occupancy_threshold,
        min_component_voxels=args.min_component_voxels,
        connectivity=args.connectivity,
        layers_df=layers_df,
        normals=normals_0,
        reference_patch_stats=ref_stats_0,
    )
    write_csv_utf8(round1_patch_df, output_dir / "round1_patches.csv")
    occ_1, normals_1, _ = rasterize_patches_to_voxel(
        patch_df=round1_patch_df,
        grid=grid,
        thickness_vox=args.thickness_vox,
        include_normals=include_normals,
    )

    round1_metrics, round1_pairs = build_chinese_metric_summary(
        before_patches=input_patch_df,
        after_patches=round1_patch_df,
        before_occupancy=occ_0,
        after_occupancy=occ_1,
        grid=grid,
        occupancy_threshold=args.occupancy_threshold,
    )
    round1_record = {
        "单元ID": unit_id,
        "转化轮次": "第一轮",
        "数据模式": str(unit_summary.get("DataMode", "")),
        "可靠性类别": str(unit_summary.get("ReliabilityClass", "")),
        **round1_metrics,
        "是否达到稳定状态": "",
    }
    write_json(
        output_dir / "round1_match_pairs.json",
        {
            "unit_id": unit_id,
            "转化轮次": "第一轮",
            "匹配裂缝对": round1_pairs,
            "指标摘要": round1_record,
        },
    )

    ref_stats_1 = summarize_patch_statistics(round1_patch_df)
    round2_patch_df, _ = fit_voxel_components_to_patches(
        occupancy=occ_1,
        grid=grid,
        threshold=args.occupancy_threshold,
        min_component_voxels=args.min_component_voxels,
        connectivity=args.connectivity,
        layers_df=layers_df,
        normals=normals_1,
        reference_patch_stats=ref_stats_1,
    )
    write_csv_utf8(round2_patch_df, output_dir / "round2_patches.csv")
    occ_2, _, _ = rasterize_patches_to_voxel(
        patch_df=round2_patch_df,
        grid=grid,
        thickness_vox=args.thickness_vox,
        include_normals=False,
    )

    round2_metrics, round2_pairs = build_chinese_metric_summary(
        before_patches=round1_patch_df,
        after_patches=round2_patch_df,
        before_occupancy=occ_1,
        after_occupancy=occ_2,
        grid=grid,
        occupancy_threshold=args.occupancy_threshold,
    )
    round2_record = {
        "单元ID": unit_id,
        "转化轮次": "第二轮",
        "数据模式": str(unit_summary.get("DataMode", "")),
        "可靠性类别": str(unit_summary.get("ReliabilityClass", "")),
        **round2_metrics,
        "是否达到稳定状态": "是" if is_second_pass_stable(round2_metrics) else "否",
    }
    write_json(
        output_dir / "round2_match_pairs.json",
        {
            "unit_id": unit_id,
            "转化轮次": "第二轮",
            "匹配裂缝对": round2_pairs,
            "指标摘要": round2_record,
        },
    )

    write_json(
        output_dir / "two_pass_summary.json",
        {
            "unit_id": unit_id,
            "patch_csv": str(patch_csv),
            "grid": grid.to_json_dict(),
            "数据模式": str(unit_summary.get("DataMode", "")),
            "可靠性类别": str(unit_summary.get("ReliabilityClass", "")),
            "第一轮指标": round1_record,
            "第二轮指标": round2_record,
        },
    )
    return [round1_record, round2_record]


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root) / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)

    screening_df = build_unit_screening(args.unit_dfn_root)
    if screening_df.empty:
        raise ValueError(f"未在 {args.unit_dfn_root} 找到可扫描的单元目录。")
    write_csv_utf8(screening_df, output_root / "单元筛选结果.csv")

    recommended_df = pick_representative_units(screening_df)
    write_csv_utf8(recommended_df, output_root / "推荐统计单元.csv")

    if args.unit_ids:
        selected_units = [str(unit_id).strip() for unit_id in args.unit_ids if str(unit_id).strip()]
    else:
        selected_units = recommended_df["单元ID"].tolist()[: int(args.max_scan_count)]
    if not selected_units:
        raise ValueError("没有选出可执行统计的单元。请检查筛选条件或手动传入 --unit-ids。")

    metrics_rows: list[dict[str, Any]] = []
    for unit_id in selected_units:
        unit_dir = Path(args.unit_dfn_root) / unit_id
        if not unit_dir.exists():
            raise FileNotFoundError(f"单元目录不存在: {unit_dir}")
        unit_output_dir = output_root / unit_id
        unit_output_dir.mkdir(parents=True, exist_ok=True)
        metrics_rows.extend(
            execute_two_pass_roundtrip_for_unit(
                unit_id=unit_id,
                unit_dir=unit_dir,
                args=args,
                output_dir=unit_output_dir,
            )
        )

    metrics_df = pd.DataFrame(metrics_rows)
    write_csv_utf8(metrics_df, output_root / "双轮转化八项指标汇总.csv")
    write_json(
        output_root / "本次统计说明.json",
        {
            "unit_dfn_root": str(args.unit_dfn_root),
            "selected_units": selected_units,
            "xy_resolution": int(args.xy_resolution),
            "z_step_ms": float(args.z_step_ms),
            "z_padding_ms": float(args.z_padding_ms),
            "thickness_vox": float(args.thickness_vox),
            "channels": str(args.channels),
            "occupancy_threshold": float(args.occupancy_threshold),
            "min_component_voxels": int(args.min_component_voxels),
            "connectivity": int(args.connectivity),
            "说明": [
                "第一轮：原始单元DFN -> 体素 -> DFN。",
                "第二轮：第一轮回转DFN -> 体素 -> DFN。",
                "第二轮的八项指标若接近完全一致，说明第一轮得到的规范化DFN表示已基本稳定。",
            ],
        },
    )

    print(f"output_root: {output_root}")
    print(f"selected_units: {', '.join(selected_units)}")
    print(f"screening_csv: {output_root / '单元筛选结果.csv'}")
    print(f"recommended_csv: {output_root / '推荐统计单元.csv'}")
    print(f"metrics_csv: {output_root / '双轮转化八项指标汇总.csv'}")


if __name__ == "__main__":
    main()
