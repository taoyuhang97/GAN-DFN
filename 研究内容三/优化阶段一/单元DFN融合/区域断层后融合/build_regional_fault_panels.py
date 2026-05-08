# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from fault_postfusion_common import (
    DEFAULT_DOCX_PATH,
    append_lines_to_docx,
    azimuth_diff_deg,
    discover_fault_patch_files,
    fit_plane_from_points,
    load_fault_patch_polydata,
    make_patch_row_from_panel_row,
    parse_fault_patch_file_info,
    is_fault_patch_in_region,
    write_csv_utf8,
    write_df_to_regional_vtk,
    write_json,
)


def emit_fault_panel_progress(stage: str, detail: str | None = None) -> None:
    if detail:
        print(f"[fault-panels] {stage} | {detail}", flush=True)
    else:
        print(f"[fault-panels] {stage}", flush=True)


@dataclass
class UnionFind:
    size: int

    def __post_init__(self) -> None:
        self.parent = list(range(self.size))
        self.rank = [0] * self.size

    def find(self, idx: int) -> int:
        root = idx
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[idx] != idx:
            nxt = self.parent[idx]
            self.parent[idx] = root
            idx = nxt
        return root

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
            return
        if self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
            return
        self.parent[root_right] = root_left
        self.rank[root_left] += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build regional fault panels directly from fault patch root directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fault-patches-root", type=Path, required=True)
    parser.add_argument("--block-x-start", type=int, required=True)
    parser.add_argument("--block-x-end", type=int, required=True)
    parser.add_argument("--block-y-start", type=int, required=True)
    parser.add_argument("--block-y-end", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=f"regional_fault_panels_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--panel-merge-xy", type=float, default=220.0)
    parser.add_argument("--panel-merge-time", type=float, default=35.0)
    parser.add_argument("--panel-strike-tol", type=float, default=20.0)
    parser.add_argument("--panel-dip-tol", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    return parser


def collect_region_fault_patch_paths(
    fault_patches_root: Path,
    block_x_start: int,
    block_x_end: int,
    block_y_start: int,
    block_y_end: int,
) -> list[Path]:
    all_patch_paths = discover_fault_patch_files(Path(fault_patches_root))
    selected_paths = [
        path
        for path in all_patch_paths
        if is_fault_patch_in_region(path, block_x_start, block_x_end, block_y_start, block_y_end)
    ]
    return selected_paths


def load_fault_patch_records(
    patch_paths: list[Path],
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total_count = len(patch_paths)
    emit_step = max(1, total_count // 20) if total_count > 0 else 1
    for patch_idx, patch_path in enumerate(patch_paths, start=1):
        info = parse_fault_patch_file_info(patch_path)
        vtp_path = Path(patch_path)
        poly = load_fault_patch_polydata(vtp_path)
        points = np.asarray(poly["points"], dtype=float)
        record = {
            "UnitID": f"BX{int(info['cell_i'])}_BY{int(info['cell_j'])}",
            "BlockX": int(info["cell_i"]),
            "BlockY": int(info["cell_j"]),
            "FaultName": str(info["fault_name"]),
            "FaultPatchFile": str(vtp_path.name),
            "FaultPatchPath": str(vtp_path),
            "CenterX": float(poly["center"][0]),
            "CenterY": float(poly["center"][1]),
            "CenterTIME": float(poly["center"][2]),
            "Area3D": float(poly["area"]),
            "BBoxXMin": float(poly["bounds"][0]),
            "BBoxXMax": float(poly["bounds"][1]),
            "BBoxYMin": float(poly["bounds"][2]),
            "BBoxYMax": float(poly["bounds"][3]),
            "BBoxZMin": float(poly["bounds"][4]),
            "BBoxZMax": float(poly["bounds"][5]),
            "StrikeDeg": float(poly["strike_deg"]),
            "DipDeg": float(poly["dip_deg"]),
            "Points": points,
        }
        records.append(record)
        if progress_hook is not None and (
            patch_idx == 1 or patch_idx == total_count or patch_idx % emit_step == 0
        ):
            progress_hook("读取断层 patch 进度", f"{patch_idx}/{total_count}, last={vtp_path.name}")
    return records


def cluster_fault_patches_to_panels(
    patch_df: pd.DataFrame,
    merge_xy: float,
    merge_time: float,
    strike_tol: float,
    dip_tol: float,
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> pd.DataFrame:
    if patch_df.empty:
        patch_df = patch_df.copy()
        patch_df["FaultPanelID"] = []
        return patch_df

    output_frames: list[pd.DataFrame] = []
    panel_counter = 0
    grouped_faults = list(patch_df.groupby("FaultName", sort=False))
    total_fault_count = len(grouped_faults)
    emit_step = max(1, total_fault_count // 10) if total_fault_count > 0 else 1
    for fault_idx, (fault_name, group) in enumerate(grouped_faults, start=1):
        group = group.copy().reset_index(drop=True)
        if len(group) == 1:
            group["FaultPanelID"] = panel_counter
            panel_counter += 1
            output_frames.append(group)
            continue
        features = np.column_stack(
            [
                group["CenterX"].to_numpy(dtype=float) / max(float(merge_xy), 1e-6),
                group["CenterY"].to_numpy(dtype=float) / max(float(merge_xy), 1e-6),
                group["CenterTIME"].to_numpy(dtype=float) / max(float(merge_time), 1e-6),
            ]
        )
        tree = cKDTree(features)
        union_find = UnionFind(len(group))
        for idx in range(len(group)):
            for other_idx in tree.query_ball_point(features[idx], r=1.75):
                if int(other_idx) <= int(idx):
                    continue
                if azimuth_diff_deg(group.at[idx, "StrikeDeg"], group.at[other_idx, "StrikeDeg"]) > float(strike_tol):
                    continue
                if abs(float(group.at[idx, "DipDeg"]) - float(group.at[other_idx, "DipDeg"])) > float(dip_tol):
                    continue
                union_find.union(idx, int(other_idx))
        root_to_panel_id: dict[int, int] = {}
        panel_ids: list[int] = []
        for idx in range(len(group)):
            root = union_find.find(idx)
            if root not in root_to_panel_id:
                root_to_panel_id[root] = panel_counter
                panel_counter += 1
            panel_ids.append(root_to_panel_id[root])
        group["FaultPanelID"] = panel_ids
        output_frames.append(group)
        if progress_hook is not None and (
            fault_idx == 1 or fault_idx == total_fault_count or fault_idx % emit_step == 0
        ):
            progress_hook(
                "FaultName 聚类进度",
                f"{fault_idx}/{total_fault_count}, fault_name={fault_name}, panel_count_so_far={panel_counter}",
            )
    return pd.concat(output_frames, ignore_index=True, sort=False)


def aggregate_fault_panels(
    clustered_df: pd.DataFrame,
    progress_hook: Callable[[str, str | None], None] | None = None,
) -> pd.DataFrame:
    if clustered_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped_panels = list(clustered_df.groupby("FaultPanelID", sort=True))
    total_panel_count = len(grouped_panels)
    emit_step = max(1, total_panel_count // 20) if total_panel_count > 0 else 1
    for panel_idx, (fault_panel_id, group) in enumerate(grouped_panels, start=1):
        all_points = np.vstack(group["Points"].tolist())
        plane = fit_plane_from_points(all_points)
        panel_vertices = np.asarray(plane["panel_vertices"], dtype=float)
        unit_ids = sorted(group["UnitID"].astype(str).unique().tolist())
        row = {
            "FaultPanelID": int(fault_panel_id),
            "FaultName": str(group["FaultName"].iloc[0]),
            "SourcePatchCount": int(len(group)),
            "SourceUnitCount": int(len(unit_ids)),
            "SourceUnitIDs": ",".join(unit_ids),
            "CenterX": float(plane["center"][0]),
            "CenterY": float(plane["center"][1]),
            "CenterTIME": float(plane["center"][2]),
            "StrikeDeg": float(plane["strike_deg"]),
            "DipDeg": float(plane["dip_deg"]),
            "PanelLength": float(max(plane["panel_length"], 1.0)),
            "PanelHeight": float(max(plane["panel_height"], 1.0)),
            "PanelArea": float(group["Area3D"].sum()),
            "NormalX": float(plane["normal"][0]),
            "NormalY": float(plane["normal"][1]),
            "NormalZ": float(plane["normal"][2]),
            "StrikeVecX": float(plane["strike_vec"][0]),
            "StrikeVecY": float(plane["strike_vec"][1]),
            "StrikeVecZ": float(plane["strike_vec"][2]),
            "DipVecX": float(plane["dip_vec"][0]),
            "DipVecY": float(plane["dip_vec"][1]),
            "DipVecZ": float(plane["dip_vec"][2]),
            "BBoxXMin": float(panel_vertices[:, 0].min()),
            "BBoxXMax": float(panel_vertices[:, 0].max()),
            "BBoxYMin": float(panel_vertices[:, 1].min()),
            "BBoxYMax": float(panel_vertices[:, 1].max()),
            "BBoxZMin": float(panel_vertices[:, 2].min()),
            "BBoxZMax": float(panel_vertices[:, 2].max()),
        }
        for vertex_idx in range(1, 5):
            row[f"V{vertex_idx}X"] = float(panel_vertices[vertex_idx - 1, 0])
            row[f"V{vertex_idx}Y"] = float(panel_vertices[vertex_idx - 1, 1])
            row[f"V{vertex_idx}Z"] = float(panel_vertices[vertex_idx - 1, 2])
        rows.append(row)
        if progress_hook is not None and (
            panel_idx == 1 or panel_idx == total_panel_count or panel_idx % emit_step == 0
        ):
            progress_hook(
                "FaultPanel 聚合进度",
                f"{panel_idx}/{total_panel_count}, fault_panel_id={fault_panel_id}",
            )
    panel_df = pd.DataFrame(rows)
    ordered_cols = [
        "FaultPanelID",
        "FaultName",
        "SourcePatchCount",
        "SourceUnitCount",
        "SourceUnitIDs",
        "CenterX",
        "CenterY",
        "CenterTIME",
        "StrikeDeg",
        "DipDeg",
        "PanelLength",
        "PanelHeight",
        "PanelArea",
        "NormalX",
        "NormalY",
        "NormalZ",
        "StrikeVecX",
        "StrikeVecY",
        "StrikeVecZ",
        "DipVecX",
        "DipVecY",
        "DipVecZ",
        "BBoxXMin",
        "BBoxXMax",
        "BBoxYMin",
        "BBoxYMax",
        "BBoxZMin",
        "BBoxZMax",
    ] + [f"V{vertex_idx}{axis}" for vertex_idx in range(1, 5) for axis in ("X", "Y", "Z")]
    return panel_df.loc[:, ordered_cols]


def export_regional_fault_panels(panel_df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_csv = output_dir / "regional_fault_panels.csv"
    panel_vtk = output_dir / "regional_fault_panels_raw.vtk"
    patch_vtk_df_rows = []
    for _, row in panel_df.iterrows():
        patch_vtk_df_rows.append(
            make_patch_row_from_panel_row(
                row,
                extra={
                    "FaultPanelID": int(row["FaultPanelID"]),
                    "SourcePatchCount": int(row["SourcePatchCount"]),
                    "SourceUnitCount": int(row["SourceUnitCount"]),
                },
            )
        )
    vtk_df = pd.DataFrame(patch_vtk_df_rows)
    write_csv_utf8(panel_df, panel_csv)
    write_df_to_regional_vtk(vtk_df, "regional_fault_panels", panel_vtk, {})
    return {"panel_csv": panel_csv, "panel_vtk": panel_vtk}


def run_build_regional_fault_panels(
    fault_patches_root: Path,
    block_x_start: int,
    block_x_end: int,
    block_y_start: int,
    block_y_end: int,
    output_root: Path,
    run_name: str,
    panel_merge_xy: float,
    panel_merge_time: float,
    panel_strike_tol: float,
    panel_dip_tol: float,
) -> dict[str, Any]:
    run_dir = Path(output_root) / str(run_name)
    emit_fault_panel_progress(
        "开始构建区域断层 panel",
        f"region=BX{min(block_x_start, block_x_end)}-{max(block_x_start, block_x_end)}, BY{min(block_y_start, block_y_end)}-{max(block_y_start, block_y_end)}",
    )
    patch_paths = collect_region_fault_patch_paths(
        fault_patches_root=Path(fault_patches_root),
        block_x_start=int(block_x_start),
        block_x_end=int(block_x_end),
        block_y_start=int(block_y_start),
        block_y_end=int(block_y_end),
    )
    emit_fault_panel_progress("完成目标断层 patch 收集", f"selected_patch_file_count={len(patch_paths)}")
    patch_records = load_fault_patch_records(patch_paths, progress_hook=emit_fault_panel_progress)
    patch_df = pd.DataFrame(patch_records)
    emit_fault_panel_progress("断层 patch 读取完成", f"fault_patch_count={len(patch_df)}")
    clustered_df = cluster_fault_patches_to_panels(
        patch_df=patch_df,
        merge_xy=float(panel_merge_xy),
        merge_time=float(panel_merge_time),
        strike_tol=float(panel_strike_tol),
        dip_tol=float(panel_dip_tol),
        progress_hook=emit_fault_panel_progress,
    )
    emit_fault_panel_progress("patch 聚类完成", f"clustered_patch_count={len(clustered_df)}")
    panel_df = aggregate_fault_panels(clustered_df, progress_hook=emit_fault_panel_progress)
    emit_fault_panel_progress("panel 聚合完成", f"fault_panel_count={len(panel_df)}")
    output_paths = export_regional_fault_panels(panel_df, run_dir)
    emit_fault_panel_progress("panel 导出完成", f"panel_vtk={output_paths['panel_vtk']}")
    summary = {
        "run_name": str(run_name),
        "fault_patches_root": str(fault_patches_root),
        "selected_patch_file_count": int(len(patch_paths)),
        "fault_patch_count": int(len(patch_df)),
        "fault_panel_count": int(len(panel_df)),
        "panel_csv": str(output_paths["panel_csv"]),
        "panel_vtk": str(output_paths["panel_vtk"]),
    }
    summary_path = run_dir / "regional_fault_panels_summary.json"
    write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.output_root) / str(args.run_name)
    if run_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(f"output run dir already exists: {run_dir}")
    summary = run_build_regional_fault_panels(
        fault_patches_root=Path(args.fault_patches_root),
        block_x_start=int(args.block_x_start),
        block_x_end=int(args.block_x_end),
        block_y_start=int(args.block_y_start),
        block_y_end=int(args.block_y_end),
        output_root=Path(args.output_root),
        run_name=str(args.run_name),
        panel_merge_xy=float(args.panel_merge_xy),
        panel_merge_time=float(args.panel_merge_time),
        panel_strike_tol=float(args.panel_strike_tol),
        panel_dip_tol=float(args.panel_dip_tol),
    )
    append_lines_to_docx(
        docx_path=args.docx_path,
        title=f"regional fault panels {args.run_name}",
        lines=[
            f"fault_patches_root: {args.fault_patches_root}",
            f"block_x_range: {min(args.block_x_start, args.block_x_end)}-{max(args.block_x_start, args.block_x_end)}",
            f"block_y_range: {min(args.block_y_start, args.block_y_end)}-{max(args.block_y_start, args.block_y_end)}",
            f"selected_patch_file_count: {summary['selected_patch_file_count']}",
            f"fault_patch_count: {summary['fault_patch_count']}",
            f"fault_panel_count: {summary['fault_panel_count']}",
            f"panel_csv: {summary['panel_csv']}",
            f"panel_vtk: {summary['panel_vtk']}",
        ],
    )
    print(f"panel_csv: {summary['panel_csv']}")
    print(f"panel_vtk: {summary['panel_vtk']}")
    print(f"fault_panel_count: {summary['fault_panel_count']}")


if __name__ == "__main__":
    main()
