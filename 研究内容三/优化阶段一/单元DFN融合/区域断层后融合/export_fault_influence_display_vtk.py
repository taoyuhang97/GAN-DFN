# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fault_postfusion_common import (
    DEFAULT_DOCX_PATH,
    append_lines_to_docx,
    read_regional_vtk_to_df,
    write_csv_utf8,
    write_df_to_regional_vtk,
    write_json,
)
from fuse_faults_into_regional_dfn_v2 import assign_fault_influence, load_fault_panels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a VTK for fault surfaces and fault-influenced fractures for visualization."
    )
    parser.add_argument("--input-vtk", type=Path, required=True, help="Original regional DFN before fault fusion.")
    parser.add_argument("--fault-panel-csv", type=Path, required=True)
    parser.add_argument("--fault-surface-vtk", type=Path, required=True)
    parser.add_argument("--fused-fractures-csv", type=Path, required=True, help="Fracture CSV after fault fusion.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--run-name",
        type=str,
        default=f"fault_influence_display_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--fault-half-band-ms", type=float, default=100.0)
    parser.add_argument("--fault-remove-ms", type=float, default=50.0)
    parser.add_argument("--fault-transition-ms", type=float, default=100.0)
    parser.add_argument("--panel-xy-buffer", type=float, default=180.0)
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_fused_induced_fractures(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "PatchOriginText" not in df.columns:
        raise ValueError(f"missing PatchOriginText in fused fracture csv: {path}")
    induced_df = df[df["PatchOriginText"].astype(str).isin(["fault_parallel", "fault_perpendicular"])].copy()
    return induced_df.reset_index(drop=True)


def load_fault_surface_df(path: Path) -> pd.DataFrame:
    df, _, _ = read_regional_vtk_to_df(Path(path))
    return df.reset_index(drop=True)


def build_influenced_original_df(
    input_vtk: Path,
    fault_panel_csv: Path,
    fault_half_band_ms: float,
    fault_remove_ms: float,
    fault_transition_ms: float,
    panel_xy_buffer: float,
) -> pd.DataFrame:
    original_df, _, _ = read_regional_vtk_to_df(Path(input_vtk))
    panel_df = load_fault_panels(Path(fault_panel_csv))
    influenced_df = assign_fault_influence(
        df=original_df,
        panel_df=panel_df,
        fault_half_band_ms=float(fault_half_band_ms),
        fault_remove_ms=float(fault_remove_ms),
        fault_transition_ms=float(fault_transition_ms),
        panel_xy_buffer=float(panel_xy_buffer),
    )
    influenced_mask = pd.to_numeric(
        influenced_df.get("FaultInfluenceWeight", 0.0), errors="coerce"
    ).fillna(0.0) > 0.0
    return influenced_df.loc[influenced_mask].copy().reset_index(drop=True)


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True, sort=False)


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.output_root) / str(args.run_name)
    if run_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(f"output run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    influenced_original_df = build_influenced_original_df(
        input_vtk=Path(args.input_vtk),
        fault_panel_csv=Path(args.fault_panel_csv),
        fault_half_band_ms=float(args.fault_half_band_ms),
        fault_remove_ms=float(args.fault_remove_ms),
        fault_transition_ms=float(args.fault_transition_ms),
        panel_xy_buffer=float(args.panel_xy_buffer),
    )
    induced_df = load_fused_induced_fractures(Path(args.fused_fractures_csv))
    fault_surface_df = load_fault_surface_df(Path(args.fault_surface_vtk))

    display_df = concat_frames([influenced_original_df, induced_df, fault_surface_df])
    if display_df.empty:
        raise ValueError("display dataframe is empty")

    output_csv = run_dir / "fault_and_influence_display.csv"
    output_vtk = run_dir / "fault_and_influence_display_raw.vtk"
    write_csv_utf8(display_df, output_csv)
    write_df_to_regional_vtk(display_df, "fault_and_influence_display", output_vtk, {})

    action_counts = (
        influenced_original_df["FaultActionText"].astype(str).value_counts(dropna=False).to_dict()
        if not influenced_original_df.empty and "FaultActionText" in influenced_original_df.columns
        else {}
    )
    origin_counts = (
        display_df["PatchOriginText"].astype(str).value_counts(dropna=False).to_dict()
        if "PatchOriginText" in display_df.columns
        else {}
    )
    summary = {
        "input_vtk": str(args.input_vtk),
        "fault_panel_csv": str(args.fault_panel_csv),
        "fault_surface_vtk": str(args.fault_surface_vtk),
        "fused_fractures_csv": str(args.fused_fractures_csv),
        "fault_half_band_ms": float(args.fault_half_band_ms),
        "fault_remove_ms": float(args.fault_remove_ms),
        "fault_transition_ms": float(args.fault_transition_ms),
        "panel_xy_buffer": float(args.panel_xy_buffer),
        "influenced_original_count": int(len(influenced_original_df)),
        "induced_fracture_count": int(len(induced_df)),
        "fault_surface_count": int(len(fault_surface_df)),
        "display_count": int(len(display_df)),
        "influenced_action_counts": action_counts,
        "display_origin_counts": origin_counts,
        "output_csv": str(output_csv),
        "output_vtk": str(output_vtk),
    }
    summary_path = run_dir / "fault_and_influence_display_summary.json"
    write_json(summary_path, summary)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title=f"fault influence display {args.run_name}",
        lines=[
            f"input_vtk: {args.input_vtk}",
            f"fault_panel_csv: {args.fault_panel_csv}",
            f"fault_surface_vtk: {args.fault_surface_vtk}",
            f"fused_fractures_csv: {args.fused_fractures_csv}",
            f"influenced_original_count: {len(influenced_original_df)}",
            f"induced_fracture_count: {len(induced_df)}",
            f"fault_surface_count: {len(fault_surface_df)}",
            f"display_count: {len(display_df)}",
            f"influenced_action_counts: {action_counts}",
            f"display_origin_counts: {origin_counts}",
            f"output_vtk: {output_vtk}",
            f"summary_json: {summary_path}",
        ],
    )
    print(f"output_vtk: {output_vtk}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
