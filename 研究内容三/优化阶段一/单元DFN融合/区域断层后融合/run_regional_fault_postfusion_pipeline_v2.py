# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from build_fault_surface_fragments_from_raw_patches import run_build_fault_surface_fragments
from build_regional_fault_panels import run_build_regional_fault_panels
from fault_postfusion_common import (
    DEFAULT_DOCX_PATH,
    append_lines_to_docx,
    write_json,
    write_legacy_vtk_polygons_preserve_patch_area,
)
from fuse_faults_into_regional_dfn_v2 import run_fault_postfusion
from merge_unit_dfn_vtks import read_legacy_vtk_polygons
from scale_merged_dfn_vtk_uniform import format_scale_suffix, scale_cell_data, scale_polygons_uniform


def emit_fault_pipeline_progress(step_idx: int, total_steps: int, stage: str, detail: str | None = None) -> None:
    prefix = f"[fault-pipeline] {step_idx}/{total_steps} {stage}"
    if detail:
        print(f"{prefix} | {detail}", flush=True)
    else:
        print(prefix, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-command fault post-fusion pipeline for an existing regional DFN VTK.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-vtk", type=Path, required=True)
    parser.add_argument("--fault-patches-root", type=Path, required=True)
    parser.add_argument("--block-x-start", type=int, required=True)
    parser.add_argument("--block-x-end", type=int, required=True)
    parser.add_argument("--block-y-start", type=int, required=True)
    parser.add_argument("--block-y-end", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=f"regional_fault_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--panel-merge-xy", type=float, default=220.0)
    parser.add_argument("--panel-merge-time", type=float, default=35.0)
    parser.add_argument("--panel-strike-tol", type=float, default=20.0)
    parser.add_argument("--panel-dip-tol", type=float, default=15.0)
    parser.add_argument("--surface-max-strike", type=float, default=70.0)
    parser.add_argument("--surface-max-dip", type=float, default=18.0)
    parser.add_argument("--surface-min-fragment-area", type=float, default=8.0)
    parser.add_argument("--surface-elongate-ratio", type=float, default=2.8)
    parser.add_argument("--surface-normal-pad", type=float, default=30.0)
    parser.add_argument("--surface-gap-ratio", type=float, default=0.95)
    parser.add_argument("--surface-display-offset-ms", type=float, default=0.6)
    parser.add_argument("--surface-max-fragment-area-ratio", type=float, default=1500.0)
    parser.add_argument("--fault-half-band-ms", type=float, default=100.0)
    parser.add_argument("--fault-remove-ms", type=float, default=50.0)
    parser.add_argument("--fault-transition-ms", type=float, default=100.0)
    parser.add_argument("--panel-xy-buffer", type=float, default=180.0)
    parser.add_argument("--parallel-ratio", type=float, default=0.65)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--final-scale-factor", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    return parser


def run_uniform_scale_step(input_vtk: Path, scale_factor: float, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = read_legacy_vtk_polygons(Path(input_vtk))
    scaled_points, scaled_polygons = scale_polygons_uniform(
        points=np.asarray(payload["points"], dtype=float),
        polygons=payload["polygons"],
        scale_factor=float(scale_factor),
    )
    scaled_cell_data = scale_cell_data(payload["cell_data"], float(scale_factor))
    if "PatchArea" in scaled_cell_data:
        scaled_cell_data["PatchArea"] = np.asarray(scaled_cell_data["PatchArea"], dtype=float) * float(scale_factor) * float(scale_factor)
    output_vtk = output_dir / f"regional_dfn_fault_embedded_scaled_x{format_scale_suffix(float(scale_factor))}.vtk"
    write_legacy_vtk_polygons_preserve_patch_area(
        path=output_vtk,
        title=f"{payload.get('title', 'DFN')}_uniform_scaled_x{format_scale_suffix(float(scale_factor))}",
        points=scaled_points,
        polygons=scaled_polygons,
        cell_data=scaled_cell_data,
        scalar_types=payload.get("scalar_types", {}),
        recompute_patch_area=True,
    )
    summary = {
        "input_vtk": str(input_vtk),
        "output_vtk": str(output_vtk),
        "scale_factor": float(scale_factor),
        "polygon_count": int(len(payload["polygons"])),
    }
    write_json(output_dir / "scale_step_summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    run_root = Path(args.output_root) / str(args.run_name)
    if run_root.exists() and not bool(args.overwrite):
        raise FileExistsError(f"output run dir already exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    if not Path(args.input_vtk).exists():
        raise FileNotFoundError(f"input VTK not found: {args.input_vtk}")

    panel_dir = run_root / "01_fault_panels"
    surface_dir = run_root / "02_fault_surface"
    fuse_dir = run_root / "03_fault_fused"
    total_steps = 4 if args.final_scale_factor is not None else 3

    emit_fault_pipeline_progress(1, total_steps, "开始 fault panels", f"output_dir={panel_dir}")
    panel_summary = run_build_regional_fault_panels(
        fault_patches_root=Path(args.fault_patches_root),
        block_x_start=int(args.block_x_start),
        block_x_end=int(args.block_x_end),
        block_y_start=int(args.block_y_start),
        block_y_end=int(args.block_y_end),
        output_root=panel_dir,
        run_name="fault_panels",
        panel_merge_xy=float(args.panel_merge_xy),
        panel_merge_time=float(args.panel_merge_time),
        panel_strike_tol=float(args.panel_strike_tol),
        panel_dip_tol=float(args.panel_dip_tol),
    )
    emit_fault_pipeline_progress(1, total_steps, "完成 fault panels", f"fault_panel_count={panel_summary['fault_panel_count']}")
    emit_fault_pipeline_progress(2, total_steps, "开始 fault surface", f"output_dir={surface_dir}")
    surface_summary = run_build_fault_surface_fragments(
        fault_patches_root=Path(args.fault_patches_root),
        block_x_start=int(args.block_x_start),
        block_x_end=int(args.block_x_end),
        block_y_start=int(args.block_y_start),
        block_y_end=int(args.block_y_end),
        output_root=surface_dir,
        run_name="fault_surface_fragments",
        surface_max_strike=float(args.surface_max_strike),
        surface_max_dip=float(args.surface_max_dip),
        surface_min_fragment_area=float(args.surface_min_fragment_area),
        surface_elongate_ratio=float(args.surface_elongate_ratio),
        surface_normal_pad=float(args.surface_normal_pad),
        surface_gap_ratio=float(args.surface_gap_ratio),
        surface_display_offset_ms=float(args.surface_display_offset_ms),
        surface_max_fragment_area_ratio=float(args.surface_max_fragment_area_ratio),
    )
    emit_fault_pipeline_progress(2, total_steps, "完成 fault surface", f"fragment_count={surface_summary['fragment_count']}")
    emit_fault_pipeline_progress(3, total_steps, "开始 fault fused", f"output_dir={fuse_dir}")
    fuse_summary = run_fault_postfusion(
        input_vtk=Path(args.input_vtk),
        fault_panel_csv=Path(panel_summary["panel_csv"]),
        fault_surface_vtk=Path(surface_summary["surface_vtk"]),
        output_root=fuse_dir,
        run_name="fault_postfusion",
        fault_half_band_ms=float(args.fault_half_band_ms),
        fault_remove_ms=float(args.fault_remove_ms),
        fault_transition_ms=float(args.fault_transition_ms),
        panel_xy_buffer=float(args.panel_xy_buffer),
        parallel_ratio=float(args.parallel_ratio),
        random_seed=int(args.random_seed),
    )
    emit_fault_pipeline_progress(3, total_steps, "完成 fault fused", f"final_polygon_count={fuse_summary['final_polygon_count']}")

    scale_summary: dict[str, Any] | None = None
    final_vtk = str(fuse_summary["output_vtk"])
    if args.final_scale_factor is not None:
        emit_fault_pipeline_progress(4, total_steps, "开始 uniform scale", f"scale_factor={float(args.final_scale_factor)}")
        scale_summary = run_uniform_scale_step(
            input_vtk=Path(fuse_summary["output_vtk"]),
            scale_factor=float(args.final_scale_factor),
            output_dir=run_root / "05_scaled",
        )
        final_vtk = str(scale_summary["output_vtk"])
        emit_fault_pipeline_progress(4, total_steps, "完成 uniform scale", f"output_vtk={final_vtk}")

    pipeline_summary = {
        "run_root": str(run_root),
        "input_vtk": str(args.input_vtk),
        "fault_patches_root": str(args.fault_patches_root),
        "fault_half_band_ms": float(args.fault_half_band_ms),
        "fault_remove_ms": float(args.fault_remove_ms),
        "fault_transition_ms": float(args.fault_transition_ms),
        "surface_max_fragment_area_ratio": float(args.surface_max_fragment_area_ratio),
        "fault_panels": panel_summary,
        "fault_surface": surface_summary,
        "fault_postfusion": fuse_summary,
        "scale": scale_summary,
        "final_vtk": final_vtk,
    }
    summary_path = run_root / "regional_fault_postfusion_pipeline_summary.json"
    write_json(summary_path, pipeline_summary)
    append_lines_to_docx(
        docx_path=args.docx_path,
        title=f"regional fault pipeline {args.run_name}",
        lines=[
            f"input_vtk: {args.input_vtk}",
            f"fault_patches_root: {args.fault_patches_root}",
            f"block_x_range: {min(args.block_x_start, args.block_x_end)}-{max(args.block_x_start, args.block_x_end)}",
            f"block_y_range: {min(args.block_y_start, args.block_y_end)}-{max(args.block_y_start, args.block_y_end)}",
            f"fault_panel_csv: {panel_summary['panel_csv']}",
            f"fault_surface_vtk: {surface_summary['surface_vtk']}",
            f"fault_fused_output: {fuse_summary['output_vtk']}",
            f"final_vtk: {final_vtk}",
            f"summary_json: {summary_path}",
        ],
    )
    print(f"final_vtk: {final_vtk}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
