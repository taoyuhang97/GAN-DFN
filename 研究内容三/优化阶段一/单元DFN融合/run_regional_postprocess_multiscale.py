# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
OPT_STAGE_DIR = THIS_DIR.parent
BASELINE_DIR = OPT_STAGE_DIR / "G_DFN监督基线"

for candidate in (THIS_DIR, BASELINE_DIR):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from baseline_common import DEFAULT_DOCX_PATH, append_lines_to_docx, write_json
from geophysical_postprocess_multiscale import (
    DEFAULT_POSTPROCESS_COMPUTE_BACKEND,
    DEFAULT_POSTPROCESS_GPU_TILE_POINTS,
    DEFAULT_POSTPROCESS_MAX_CPU_THREADS,
    postprocess_vtk,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wrap multiscale regional DFN postprocess into a run directory with stable outputs."
    )
    parser.add_argument("--input-vtk", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=f"postprocess_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--output-vtk-name", type=str, default="regional_dfn_postprocessed.vtk")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=2)
    parser.add_argument("--boundary-connect", action="store_true")
    parser.add_argument("--bc-seam-fill", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compute-backend",
        type=str,
        default=DEFAULT_POSTPROCESS_COMPUTE_BACKEND,
        choices=["auto", "cpu", "gpu"],
    )
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=DEFAULT_POSTPROCESS_MAX_CPU_THREADS,
    )
    parser.add_argument(
        "--gpu-tile-points",
        type=int,
        default=DEFAULT_POSTPROCESS_GPU_TILE_POINTS,
    )
    parser.add_argument("--gmm-bic-sample-cap", type=int, default=250000)
    parser.add_argument("--gmm-fit-sample-cap", type=int, default=400000)
    parser.add_argument("--phase2-chunk-row-threshold", type=int, default=250000)
    parser.add_argument("--phase2-chunk-unit-width", type=int, default=12)
    parser.add_argument("--phase2-chunk-unit-height", type=int, default=12)
    parser.add_argument("--phase2-chunk-overlap-units", type=int, default=1)
    parser.add_argument("--disable-phase2-chunking", action="store_true")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_vtk = Path(args.input_vtk)
    if not input_vtk.exists():
        raise FileNotFoundError(f"input_vtk not found: {input_vtk}")

    run_dir = Path(args.output_root) / str(args.run_name)
    if run_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(f"output run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    output_vtk = run_dir / str(args.output_vtk_name)
    pipeline_kwargs: dict[str, Any] = {}
    pipeline_kwargs["compute_backend"] = str(args.compute_backend)
    pipeline_kwargs["max_cpu_threads"] = int(args.max_cpu_threads)
    pipeline_kwargs["gpu_tile_points"] = int(args.gpu_tile_points)
    pipeline_kwargs["fracture_set_bic_sample_cap"] = int(args.gmm_bic_sample_cap)
    pipeline_kwargs["fracture_set_fit_sample_cap"] = int(args.gmm_fit_sample_cap)
    pipeline_kwargs["phase2_chunk_row_threshold"] = int(args.phase2_chunk_row_threshold)
    pipeline_kwargs["phase2_chunk_unit_width"] = int(args.phase2_chunk_unit_width)
    pipeline_kwargs["phase2_chunk_unit_height"] = int(args.phase2_chunk_unit_height)
    pipeline_kwargs["phase2_chunk_overlap_units"] = int(args.phase2_chunk_overlap_units)
    pipeline_kwargs["disable_phase2_chunking"] = bool(args.disable_phase2_chunking)
    if bool(args.boundary_connect):
        pipeline_kwargs["enable_boundary_connect"] = True
    if bool(args.bc_seam_fill):
        pipeline_kwargs["enable_seam_fill"] = True

    stats = postprocess_vtk(
        input_vtk=input_vtk,
        output_vtk=output_vtk,
        phase=int(args.phase),
        **pipeline_kwargs,
    )

    summary = {
        "run_dir": str(run_dir),
        "input_vtk": str(input_vtk),
        "output_vtk": str(output_vtk),
        "phase": int(args.phase),
        "boundary_connect": bool(args.boundary_connect),
        "bc_seam_fill": bool(args.bc_seam_fill),
        "compute_backend": str(args.compute_backend),
        "max_cpu_threads": int(args.max_cpu_threads),
        "gpu_tile_points": int(args.gpu_tile_points),
        "gmm_bic_sample_cap": int(args.gmm_bic_sample_cap),
        "gmm_fit_sample_cap": int(args.gmm_fit_sample_cap),
        "phase2_chunk_row_threshold": int(args.phase2_chunk_row_threshold),
        "phase2_chunk_unit_width": int(args.phase2_chunk_unit_width),
        "phase2_chunk_unit_height": int(args.phase2_chunk_unit_height),
        "phase2_chunk_overlap_units": int(args.phase2_chunk_overlap_units),
        "disable_phase2_chunking": bool(args.disable_phase2_chunking),
        "stats": stats,
    }
    summary_path = run_dir / "regional_postprocess_summary.json"
    write_json(summary_path, summary)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title=f"regional postprocess {args.run_name}",
        lines=[
            f"input_vtk: {input_vtk}",
            f"output_vtk: {output_vtk}",
            f"phase: {args.phase}",
            f"boundary_connect: {bool(args.boundary_connect)}",
            f"bc_seam_fill: {bool(args.bc_seam_fill)}",
            f"compute_backend: {args.compute_backend}",
            f"max_cpu_threads: {args.max_cpu_threads}",
            f"gpu_tile_points: {args.gpu_tile_points}",
            f"gmm_bic_sample_cap: {args.gmm_bic_sample_cap}",
            f"gmm_fit_sample_cap: {args.gmm_fit_sample_cap}",
            f"phase2_chunk_row_threshold: {args.phase2_chunk_row_threshold}",
            f"phase2_chunk_unit_width: {args.phase2_chunk_unit_width}",
            f"phase2_chunk_unit_height: {args.phase2_chunk_unit_height}",
            f"phase2_chunk_overlap_units: {args.phase2_chunk_overlap_units}",
            f"disable_phase2_chunking: {bool(args.disable_phase2_chunking)}",
            f"input_count: {stats.get('input_count', '?')}",
            f"output_count: {stats.get('output_count', '?')}",
            f"summary_json: {summary_path}",
        ],
    )
    print(f"output_vtk: {output_vtk}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
