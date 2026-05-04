# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from baseline_common import DEFAULT_LAYER_DENSITY_SOURCE, LAYER_DENSITY_SOURCE_CHOICES


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent

STATS_SCRIPT = ROOT_DIR / "GAN训练准备" / "窗口级实例统计" / "analyze_instance_window_distribution.py"
PACKAGE_SCRIPT = ROOT_DIR / "GAN训练准备" / "训练样本打包" / "build_sparse_instance_gan_dataset.py"
SPLIT_SCRIPT = THIS_DIR / "build_unit_level_split.py"
TRAIN_SCRIPT = THIS_DIR / "train_layerwise_models.py"
EVAL_SCRIPT = THIS_DIR / "infer_and_evaluate_baseline.py"
PRODUCTION_SCRIPT = THIS_DIR / "infer_production_units_baseline.py"
MERGE_SCRIPT = THIS_DIR / "merge_unit_dfn_vtks.py"
POSTPROCESS_SCRIPT = ROOT_DIR / "单元DFN融合" / "run_regional_postprocess_multiscale.py"
FAULT_POSTFUSION_SCRIPT = ROOT_DIR / "单元DFN融合" / "区域断层后融合" / "run_regional_fault_postfusion_pipeline_v2.py"

DEFAULT_PIPELINE_OUTPUT_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/G_DFN监督基线/后台全流程"
)
DEFAULT_UNIT_DFN_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/单元DFN构建/批量生成_新层位重拆分"
)
DEFAULT_TRACE_HEADER_CSV = Path(r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/trace_header_xy.csv")
DEFAULT_SGY_FILE = Path(r"/data/shared/project-oil/wx数据/砂砾岩/psdm_final_time.sgy")
DEFAULT_SURFACE_DIR = Path(r"/data/shared/project-oil/wx数据/砂砾岩/层位")
DEFAULT_DOCX_PATH = Path(r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/实验记录/实验记录20260328.docx")
DEFAULT_FAULT_PATCHES_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/研究内容二/单元实验/fault_patches_out/patches"
)
DEFAULT_INPUT_CHANNELS = [
    "seismic_amp",
    "grad_x",
    "grad_y",
    "grad_z",
    "rel_depth_in_interval",
]
DEFAULT_CANDIDATE_K = [4, 8, 12, 16, 24, 32, 40]
DEFAULT_REQUIRED_UNIT_FILES = [
    "unit_layers_input.csv",
    "predicted_unit_patches.csv",
    "predicted_patches_raw_time.vtk",
    "unit_prediction_summary.json",
]
STAGE_ORDER = [
    "stats",
    "package",
    "split",
    "train_layerwise",
    "eval",
    "smoke_infer",
    "full_infer",
    "merge",
    "postprocess",
    "fault_postfusion",
]
STAGE_DEPENDENCIES = {
    "stats": [],
    "package": ["stats"],
    "split": ["package"],
    "train_layerwise": ["split"],
    "eval": ["split", "train_layerwise"],
    "smoke_infer": ["train_layerwise"],
    "full_infer": ["train_layerwise"],
    "merge": ["full_infer"],
    "postprocess": ["merge"],
    "fault_postfusion": ["postprocess"],
}


@dataclass(frozen=True)
class PipelinePaths:
    pipeline_root: Path
    state_path: Path
    logs_dir: Path
    stats_output_root: Path
    stats_run_dir: Path
    dataset_output_root: Path
    dataset_run_dir: Path
    split_output_root: Path
    split_run_dir: Path
    train_output_root: Path
    train_registry_py: Path
    train_summary_json: Path
    eval_output_root: Path
    eval_run_dir: Path
    smoke_output_root: Path
    smoke_run_dir: Path
    full_output_root: Path
    full_run_dir: Path
    merge_output_root: Path
    merge_run_dir: Path
    merge_output_vtk: Path
    postprocess_output_root: Path
    postprocess_run_dir: Path
    postprocess_summary_json: Path
    postprocess_output_vtk: Path
    fault_output_root: Path
    fault_run_dir: Path
    fault_summary_json: Path


@dataclass
class StageResult:
    executed: bool
    detail: str


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported json value: {type(value)}")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def quote_cmd(cmd: list[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def resolve_path_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(Path(text).resolve())


def same_path(value: Any, expected: Path) -> bool:
    left = resolve_path_text(value)
    right = str(Path(expected).resolve())
    return bool(left) and left == right


def load_state(state_path: Path, run_name: str, args: argparse.Namespace) -> dict[str, Any]:
    existing = read_json(state_path)
    if existing is not None:
        return existing
    return {
        "run_name": run_name,
        "created_at": now_text(),
        "updated_at": now_text(),
        "requested_config": serialize_args(args),
        "stages": {},
    }


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_text()
    write_json(state_path, state)


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, list):
            payload[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            payload[key] = value
    return payload


def build_paths(args: argparse.Namespace) -> PipelinePaths:
    pipeline_root = Path(args.pipeline_output_root) / str(args.run_name)
    return PipelinePaths(
        pipeline_root=pipeline_root,
        state_path=pipeline_root / "pipeline_state.json",
        logs_dir=pipeline_root / "logs",
        stats_output_root=pipeline_root / "stats",
        stats_run_dir=pipeline_root / "stats" / "stats",
        dataset_output_root=pipeline_root / "dataset",
        dataset_run_dir=pipeline_root / "dataset" / "dataset",
        split_output_root=pipeline_root / "split",
        split_run_dir=pipeline_root / "split" / "split",
        train_output_root=pipeline_root / "train_layerwise",
        train_registry_py=pipeline_root / "train_layerwise" / "layer_model_registry.py",
        train_summary_json=pipeline_root / "train_layerwise" / "layerwise_baseline_layer_training_summary.json",
        eval_output_root=pipeline_root / "eval",
        eval_run_dir=pipeline_root / "eval" / "eval_test",
        smoke_output_root=pipeline_root / "production",
        smoke_run_dir=pipeline_root / "production" / "smoke_infer",
        full_output_root=pipeline_root / "production",
        full_run_dir=pipeline_root / "production" / "full_infer",
        merge_output_root=pipeline_root / "merge",
        merge_run_dir=pipeline_root / "merge" / "merge_full",
        merge_output_vtk=pipeline_root / "merge" / "merge_full" / "merged_full_infer_predicted_patches_raw_time.vtk",
        postprocess_output_root=pipeline_root / "postprocess",
        postprocess_run_dir=pipeline_root / "postprocess" / "postprocess_full",
        postprocess_summary_json=pipeline_root / "postprocess" / "postprocess_full" / "regional_postprocess_summary.json",
        postprocess_output_vtk=pipeline_root / "postprocess" / "postprocess_full" / "regional_dfn_postprocessed.vtk",
        fault_output_root=pipeline_root / "fault_postfusion",
        fault_run_dir=pipeline_root / "fault_postfusion" / "fault_postfusion_full",
        fault_summary_json=pipeline_root / "fault_postfusion" / "fault_postfusion_full" / "regional_fault_postfusion_pipeline_summary.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按固定规则后台编排分层训练、冒烟推理、全量推理与区域合并。支持校验、跳过、断点续跑和失败重试。"
    )
    parser.add_argument("--run-name", type=str, default=f"layerwise_full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--pipeline-output-root", type=Path, default=DEFAULT_PIPELINE_OUTPUT_ROOT)
    parser.add_argument("--conda-env", type=str, default="gan-dfn")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart-from-stage", choices=STAGE_ORDER)
    parser.add_argument("--stop-after", choices=STAGE_ORDER)
    parser.add_argument("--force-stage", nargs="+", choices=STAGE_ORDER, default=[])
    parser.add_argument("--max-stage-retries", type=int, default=1)
    parser.add_argument("--max-shard-retries", type=int, default=1)
    parser.add_argument("--retry-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)

    parser.add_argument("--unit-dfn-root", type=Path, default=DEFAULT_UNIT_DFN_ROOT)
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--sgy-file", type=Path, default=DEFAULT_SGY_FILE)
    parser.add_argument("--surface-dir", type=Path, default=DEFAULT_SURFACE_DIR)

    parser.add_argument("--xy-resolution", type=int, default=24)
    parser.add_argument("--z-step-ms", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--candidate-k", type=int, nargs="+", default=list(DEFAULT_CANDIDATE_K))
    parser.add_argument("--limit-units", type=int)

    parser.add_argument(
        "--input-channels",
        nargs="+",
        default=list(DEFAULT_INPUT_CHANNELS),
    )
    parser.add_argument("--input-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--geom-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--omit-count-volume", action="store_true")

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-unit-count", type=int)
    parser.add_argument("--val-unit-count", type=int)
    parser.add_argument("--test-unit-count", type=int)
    parser.add_argument("--fixed-unit-split-csv", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260330)

    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--retrain-layerwise", action="store_true")
    parser.add_argument("--slots-per-voxel", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--torch-num-interop-threads", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--train-limit-samples", type=int)
    parser.add_argument("--val-limit-samples", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-steps", type=int)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-cache-raw-packages", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--center-positive-weight", type=float, default=24.0)
    parser.add_argument("--center-negative-weight", type=float, default=0.25)
    parser.add_argument("--center-focal-gamma", type=float, default=2.0)
    parser.add_argument("--count-positive-weight", type=float, default=12.0)
    parser.add_argument("--count-negative-weight", type=float, default=0.25)
    parser.add_argument("--center-loss-weight", type=float, default=2.5)
    parser.add_argument("--count-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--calibration-center-thresholds",
        nargs="+",
        type=float,
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument(
        "--calibration-count-thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    )
    parser.add_argument("--calibration-window-weight", type=float, default=0.35)

    parser.add_argument("--eval-split-name", choices=["train", "val", "test"], default="test")
    parser.add_argument("--eval-limit-samples", type=int)
    parser.add_argument("--eval-limit-units", type=int)
    parser.add_argument("--eval-center-threshold", type=float, default=0.7)
    parser.add_argument("--eval-decode-mode", choices=["strict", "relaxed"], default="strict")
    parser.add_argument("--eval-relaxed-min-count", type=int, default=1)
    parser.add_argument("--eval-count-activation-threshold", type=float, default=0.5)
    parser.add_argument("--eval-min-count-if-active", type=int, default=1)

    parser.add_argument("--smoke-block-x-start", type=int, default=70)
    parser.add_argument("--smoke-block-x-end", type=int, default=72)
    parser.add_argument("--smoke-block-y-start", type=int, default=44)
    parser.add_argument("--smoke-block-y-end", type=int, default=46)
    parser.add_argument("--smoke-limit-units", type=int)
    parser.add_argument("--smoke-artifact-profile", choices=["compact", "standard", "debug"], default="standard")
    parser.add_argument("--skip-smoke-infer", action="store_true")

    parser.add_argument("--full-block-x-start", type=int, default=0)
    parser.add_argument("--full-block-x-end", type=int, default=84)
    parser.add_argument("--full-block-y-start", type=int, default=0)
    parser.add_argument("--full-block-y-end", type=int, default=60)
    parser.add_argument("--full-unit-shard-count", type=int)
    parser.add_argument("--artifact-profile", choices=["compact", "standard", "debug"], default="compact")
    parser.add_argument("--layer-top-boundary-ms", type=float, default=1100.0)
    parser.add_argument("--layer-bottom-boundary-ms", type=float, default=3800.0)

    parser.add_argument("--center-threshold", type=float, default=0.7)
    parser.add_argument("--patch-scale-factor", type=float, default=1.0)
    parser.add_argument("--decode-mode", choices=["strict", "relaxed"], default="strict")
    parser.add_argument("--relaxed-min-count", type=int, default=1)
    parser.add_argument("--count-activation-threshold", type=float, default=0.5)
    parser.add_argument("--min-count-if-active", type=int, default=1)
    parser.add_argument("--max-total-patches-per-window", type=int, default=256)
    parser.add_argument("--disable-layer-density-control", action="store_true")
    parser.add_argument("--layer-density-source", choices=list(LAYER_DENSITY_SOURCE_CHOICES), default=DEFAULT_LAYER_DENSITY_SOURCE)
    parser.add_argument("--layer-density-scale", type=float, default=1.0)
    parser.add_argument("--layer-density-calibration-min-scale", type=float, default=0.25)
    parser.add_argument("--layer-density-calibration-max-scale", type=float, default=4.0)
    parser.add_argument("--max-slots-per-voxel", type=int, default=16)
    parser.add_argument("--dedupe-xy-tol-m", type=float, default=6.25)
    parser.add_argument("--dedupe-time-tol-ms", type=float, default=0.4)
    parser.add_argument("--dedupe-azimuth-tol-deg", type=float, default=20.0)
    parser.add_argument("--dedupe-dip-tol-deg", type=float, default=12.0)
    parser.add_argument("--display-z-scale", type=float, default=5.0)
    parser.add_argument("--vtk-no-invert-time", action="store_true")
    parser.add_argument("--save-window-csv", action="store_true")
    parser.add_argument("--save-window-packages", action="store_true")
    parser.add_argument("--partial-save-every-windows", type=int, default=10)
    parser.add_argument("--no-layer-constraint", action="store_true")
    parser.add_argument("--disable-seismic-bound-extension", action="store_true")
    parser.add_argument("--time-min", type=float)
    parser.add_argument("--time-max", type=float)

    parser.add_argument("--merge-vtk-name", type=str, default="predicted_patches_raw_time.vtk")
    parser.add_argument("--postprocess-phase", type=int, choices=[1, 2], default=2)
    parser.add_argument("--postprocess-boundary-connect", action="store_true")
    parser.add_argument("--postprocess-bc-seam-fill", action="store_true")
    parser.add_argument("--postprocess-compute-backend", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--postprocess-max-cpu-threads", type=int, default=24)
    parser.add_argument("--postprocess-gpu-tile-points", type=int, default=2048)
    parser.add_argument("--fault-patches-root", type=Path, default=DEFAULT_FAULT_PATCHES_ROOT)
    parser.add_argument("--fault-half-band-ms", type=float, default=100.0)
    parser.add_argument("--fault-remove-ms", type=float, default=50.0)
    parser.add_argument("--fault-transition-ms", type=float, default=100.0)
    parser.add_argument("--fault-surface-max-fragment-area-ratio", type=float, default=1500.0)
    return parser


def build_conda_python_cmd(conda_env: str, script_path: Path, extra_args: list[str]) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", str(conda_env), "python", str(script_path), *extra_args]


def run_command(
    cmd: list[str],
    log_path: Path,
    cwd: Path,
    env_overrides: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        env.update({key: str(value) for key, value in env_overrides.items() if value is not None})
    with log_path.open("a", encoding="utf-8") as log_fp:
        log_fp.write(f"\n[{now_text()}] COMMAND START\n")
        log_fp.write(quote_cmd(cmd) + "\n\n")
        log_fp.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
        return_code = process.wait()
        log_fp.write(f"\n[{now_text()}] COMMAND END return_code={return_code}\n")
        log_fp.flush()
    return int(return_code)


def ensure_scripts_exist() -> None:
    for path in [
        STATS_SCRIPT,
        PACKAGE_SCRIPT,
        SPLIT_SCRIPT,
        TRAIN_SCRIPT,
        EVAL_SCRIPT,
        PRODUCTION_SCRIPT,
        MERGE_SCRIPT,
        POSTPROCESS_SCRIPT,
        FAULT_POSTFUSION_SCRIPT,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"required script not found: {path}")


def collect_rect_unit_ids(
    block_x_start: int,
    block_x_end: int,
    block_y_start: int,
    block_y_end: int,
    limit_units: int | None = None,
) -> list[str]:
    start_x = min(int(block_x_start), int(block_x_end))
    end_x = max(int(block_x_start), int(block_x_end))
    start_y = min(int(block_y_start), int(block_y_end))
    end_y = max(int(block_y_start), int(block_y_end))
    unit_ids = [
        f"BX{block_x}_BY{block_y}"
        for block_x in range(start_x, end_x + 1)
        for block_y in range(start_y, end_y + 1)
    ]
    if limit_units is not None:
        return unit_ids[: int(limit_units)]
    return unit_ids


def shard_unit_ids(unit_ids: list[str], shard_index: int, shard_count: int) -> list[str]:
    if int(shard_count) <= 1:
        return list(unit_ids)
    return [unit_id for idx, unit_id in enumerate(unit_ids) if idx % int(shard_count) == int(shard_index)]


def build_shard_suffix(shard_index: int, shard_count: int) -> str:
    if int(shard_count) <= 1:
        return ""
    return f"_shard_{int(shard_index):02d}of{int(shard_count):02d}"


def read_csv_unit_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            return []
        first_field = reader.fieldnames[0]
        unit_ids: list[str] = []
        for row in reader:
            value = str(row.get("UnitID", "") or row.get(first_field, "")).strip()
            if value:
                unit_ids.append(value)
        return unit_ids


def wait_for_existing_files(
    expected_paths: list[Path],
    attempts: int = 5,
    sleep_seconds: float = 1.0,
) -> list[Path]:
    pending = [Path(path) for path in expected_paths]
    for attempt in range(max(1, int(attempts))):
        missing = [path for path in pending if not path.exists()]
        if not missing:
            return []
        if attempt < max(1, int(attempts)) - 1:
            time.sleep(max(0.0, float(sleep_seconds)))
        pending = missing
    return pending


def load_registry_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"_layer_registry_{path.stem}", str(path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    registry = (
        getattr(module, "LAYER_MODEL_REGISTRY", None)
        or getattr(module, "MODEL_REGISTRY", None)
        or getattr(module, "REGISTRY", None)
    )
    if not isinstance(registry, dict):
        return None
    if "models" in registry and isinstance(registry["models"], dict):
        return registry
    return {"models": registry}


def validate_stats(paths: PipelinePaths, args: argparse.Namespace) -> tuple[bool, str]:
    summary_path = paths.stats_run_dir / "aggregated" / "global_summary.json"
    payload = read_json(summary_path)
    if payload is None:
        return False, f"missing_or_invalid_json: {summary_path}"
    if not same_path(payload.get("unit_dfn_root", ""), Path(args.unit_dfn_root)):
        return False, "unit_dfn_root_mismatch"
    if not same_path(payload.get("trace_header_csv", ""), Path(args.trace_header_csv)):
        return False, "trace_header_csv_mismatch"
    if int(payload.get("xy_resolution", -1)) != int(args.xy_resolution):
        return False, "xy_resolution_mismatch"
    if float(payload.get("z_step_ms", -1.0)) != float(args.z_step_ms):
        return False, "z_step_ms_mismatch"
    payload_window_sizes = [int(v) for v in payload.get("window_sizes", [])]
    if payload_window_sizes != [int(args.window_size)]:
        return False, "window_sizes_mismatch"
    if float(payload.get("overlap_ratio", -1.0)) != float(args.overlap_ratio):
        return False, "overlap_ratio_mismatch"
    payload_candidate_k = [int(v) for v in payload.get("candidate_k", [])]
    if payload_candidate_k != [int(v) for v in args.candidate_k]:
        return False, "candidate_k_mismatch"
    processed = int(payload.get("processed_unit_count", 0))
    if processed <= 0:
        return False, "processed_unit_count<=0"
    return True, f"processed_unit_count={processed}"


def validate_package(paths: PipelinePaths, args: argparse.Namespace) -> tuple[bool, str]:
    summary_path = paths.dataset_run_dir / "aggregated" / "dataset_summary.json"
    manifest_path = paths.dataset_run_dir / "aggregated" / "sample_manifest.csv"
    payload = read_json(summary_path)
    if payload is None:
        return False, f"missing_or_invalid_json: {summary_path}"
    if not manifest_path.exists():
        return False, f"missing_manifest: {manifest_path}"
    if not same_path(payload.get("stats_run_dir", ""), paths.stats_run_dir):
        return False, "stats_run_dir_mismatch"
    if not same_path(payload.get("sgy_file", ""), Path(args.sgy_file)):
        return False, "sgy_file_mismatch"
    if list(payload.get("input_channels", [])) != [str(v) for v in args.input_channels]:
        return False, "input_channels_mismatch"
    if str(payload.get("input_dtype", "")) != str(args.input_dtype):
        return False, "input_dtype_mismatch"
    if str(payload.get("geom_dtype", "")) != str(args.geom_dtype):
        return False, "geom_dtype_mismatch"
    if bool(payload.get("count_volume_included", False)) != bool(not args.omit_count_volume):
        return False, "count_volume_flag_mismatch"
    total_sample_count = int(payload.get("total_sample_count", 0))
    if total_sample_count <= 0:
        return False, "total_sample_count<=0"
    return True, f"total_sample_count={total_sample_count}"


def validate_split(paths: PipelinePaths, args: argparse.Namespace) -> tuple[bool, str]:
    summary_path = paths.split_run_dir / "split_summary.json"
    payload = read_json(summary_path)
    if payload is None:
        return False, f"missing_or_invalid_json: {summary_path}"
    for split_name in ("train", "val", "test"):
        manifest_path = paths.split_run_dir / f"{split_name}_manifest.csv"
        if not manifest_path.exists():
            return False, f"missing_manifest: {manifest_path}"
    if not same_path(payload.get("dataset_run_dir", ""), paths.dataset_run_dir):
        return False, "dataset_run_dir_mismatch"
    if int(payload.get("seed", -1)) != int(args.split_seed):
        return False, "seed_mismatch"
    if float(payload.get("train_ratio", -1.0)) != float(args.train_ratio):
        return False, "train_ratio_mismatch"
    if float(payload.get("val_ratio", -1.0)) != float(args.val_ratio):
        return False, "val_ratio_mismatch"
    expected_train_unit_count = int(args.train_unit_count) if args.train_unit_count is not None else None
    expected_val_unit_count = int(args.val_unit_count) if args.val_unit_count is not None else None
    expected_test_unit_count = int(args.test_unit_count) if args.test_unit_count is not None else None
    if payload.get("train_unit_count_requested") != expected_train_unit_count:
        return False, "train_unit_count_requested_mismatch"
    if payload.get("val_unit_count_requested") != expected_val_unit_count:
        return False, "val_unit_count_requested_mismatch"
    if payload.get("test_unit_count_requested") != expected_test_unit_count:
        return False, "test_unit_count_requested_mismatch"
    expected_fixed_unit_split_csv = str(Path(args.fixed_unit_split_csv).resolve()) if args.fixed_unit_split_csv else None
    if payload.get("fixed_unit_split_csv") != expected_fixed_unit_split_csv:
        return False, "fixed_unit_split_csv_mismatch"
    train_windows = int(payload.get("train_window_count", 0))
    val_windows = int(payload.get("val_window_count", 0))
    test_windows = int(payload.get("test_window_count", 0))
    if min(train_windows, val_windows, test_windows) <= 0:
        return False, "one_of_split_window_count<=0"
    return True, f"train/val/test_windows={train_windows}/{val_windows}/{test_windows}"


def validate_train(paths: PipelinePaths, args: argparse.Namespace) -> tuple[bool, str]:
    registry_payload = load_registry_payload(paths.train_registry_py)
    if registry_payload is None:
        return False, f"missing_or_invalid_registry: {paths.train_registry_py}"
    metadata = registry_payload.get("metadata", {}) if isinstance(registry_payload, dict) else {}
    if metadata and not same_path(metadata.get("split_run_dir", ""), paths.split_run_dir):
        return False, "split_run_dir_mismatch"
    if metadata and not same_path(metadata.get("unit_dfn_root", ""), Path(args.unit_dfn_root)):
        return False, "unit_dfn_root_mismatch"
    models = registry_payload.get("models", {}) if isinstance(registry_payload, dict) else {}
    if not isinstance(models, dict) or not models:
        return False, "empty_registry_models"
    missing_paths = [path for path in models.values() if not Path(str(path)).exists()]
    if missing_paths:
        return False, f"missing_checkpoint_count={len(missing_paths)}"
    layer_decode_settings = metadata.get("layer_decode_settings", {}) if isinstance(metadata, dict) else {}
    if not isinstance(layer_decode_settings, dict) or not layer_decode_settings:
        return False, "empty_layer_decode_settings"
    layer_density_priors = metadata.get("layer_density_priors", {}) if isinstance(metadata, dict) else {}
    if not isinstance(layer_density_priors, dict) or not layer_density_priors:
        return False, "empty_layer_density_priors"
    return True, (
        f"registry_model_count={len(models)}, "
        f"layer_decode_settings_count={len(layer_decode_settings)}, "
        f"layer_density_priors_count={len(layer_density_priors)}"
    )


def validate_eval(paths: PipelinePaths, args: argparse.Namespace) -> tuple[bool, str]:
    summary_path = paths.eval_run_dir / "aggregated" / "evaluation_summary.json"
    payload = read_json(summary_path)
    if payload is None:
        return False, f"missing_or_invalid_json: {summary_path}"
    expected_manifest = paths.split_run_dir / f"{args.eval_split_name}_manifest.csv"
    if not same_path(payload.get("layer_model_registry_py", ""), paths.train_registry_py):
        return False, "layer_model_registry_mismatch"
    if not same_path(payload.get("split_manifest_csv", ""), expected_manifest):
        return False, "split_manifest_mismatch"
    if str(payload.get("decode_mode", "")) != str(args.eval_decode_mode):
        return False, "decode_mode_mismatch"
    if int(payload.get("relaxed_min_count", -1)) != int(args.eval_relaxed_min_count):
        return False, "relaxed_min_count_mismatch"
    if float(payload.get("count_activation_threshold", -1.0)) != float(args.eval_count_activation_threshold):
        return False, "count_activation_threshold_mismatch"
    if int(payload.get("min_count_if_active", -1)) != int(args.eval_min_count_if_active):
        return False, "min_count_if_active_mismatch"
    if bool(payload.get("layer_density_control_enabled", False)) != bool(not args.disable_layer_density_control):
        return False, "layer_density_control_flag_mismatch"
    if str(payload.get("layer_density_source", "")) != str(args.layer_density_source):
        return False, "layer_density_source_mismatch"
    if float(payload.get("layer_density_scale", -1.0)) != float(args.layer_density_scale):
        return False, "layer_density_scale_mismatch"
    if float(payload.get("layer_density_calibration_min_scale", -1.0)) != float(args.layer_density_calibration_min_scale):
        return False, "layer_density_calibration_min_scale_mismatch"
    if float(payload.get("layer_density_calibration_max_scale", -1.0)) != float(args.layer_density_calibration_max_scale):
        return False, "layer_density_calibration_max_scale_mismatch"
    evaluated = int(payload.get("evaluated_unit_count", 0))
    if evaluated <= 0:
        return False, "evaluated_unit_count<=0"
    return True, f"evaluated_unit_count={evaluated}"


def validate_production_run(
    run_dir: Path,
    expected_unit_ids: list[str],
    shard_index: int,
    shard_count: int,
    args: argparse.Namespace,
    registry_py: Path,
    expected_artifact_profile: str,
) -> tuple[bool, str]:
    suffix = build_shard_suffix(shard_index=shard_index, shard_count=shard_count)
    aggregated_dir = run_dir / "aggregated"
    units_dir = run_dir / "units"
    summary_path = aggregated_dir / f"production_inference_summary{suffix}.json"
    selected_units_csv = aggregated_dir / f"selected_units{suffix}.csv"
    payload = read_json(summary_path)
    if payload is None:
        return False, f"missing_or_invalid_json: {summary_path}"
    if not selected_units_csv.exists():
        return False, f"missing_selected_units_csv: {selected_units_csv}"
    if not same_path(payload.get("layer_model_registry_py", ""), registry_py):
        return False, "layer_model_registry_mismatch"
    if not same_path(payload.get("unit_dfn_root", ""), Path(args.unit_dfn_root)):
        return False, "unit_dfn_root_mismatch"
    if not same_path(payload.get("surface_dir", ""), Path(args.surface_dir)):
        return False, "surface_dir_mismatch"
    if not same_path(payload.get("trace_header_csv", ""), Path(args.trace_header_csv)):
        return False, "trace_header_csv_mismatch"
    if not same_path(payload.get("sgy_file", ""), Path(args.sgy_file)):
        return False, "sgy_file_mismatch"
    if str(payload.get("artifact_profile", "")) != str(expected_artifact_profile):
        return False, "artifact_profile_mismatch"
    if int(payload.get("unit_shard_index", -1)) != int(shard_index):
        return False, "unit_shard_index_mismatch"
    if int(payload.get("unit_shard_count", -1)) != int(shard_count):
        return False, "unit_shard_count_mismatch"
    if list(payload.get("input_channels", [])) != [str(v) for v in args.input_channels]:
        return False, "input_channels_mismatch"
    if float(payload.get("center_threshold", -1.0)) != float(args.center_threshold):
        return False, "center_threshold_mismatch"
    if float(payload.get("patch_scale_factor", -1.0)) != float(args.patch_scale_factor):
        return False, "patch_scale_factor_mismatch"
    if str(payload.get("decode_mode", "")) != str(args.decode_mode):
        return False, "decode_mode_mismatch"
    if int(payload.get("relaxed_min_count", -1)) != int(args.relaxed_min_count):
        return False, "relaxed_min_count_mismatch"
    if float(payload.get("count_activation_threshold", -1.0)) != float(args.count_activation_threshold):
        return False, "count_activation_threshold_mismatch"
    if int(payload.get("min_count_if_active", -1)) != int(args.min_count_if_active):
        return False, "min_count_if_active_mismatch"
    if bool(payload.get("layer_density_control_enabled", False)) != bool(not args.disable_layer_density_control):
        return False, "layer_density_control_flag_mismatch"
    if str(payload.get("layer_density_source", "")) != str(args.layer_density_source):
        return False, "layer_density_source_mismatch"
    if float(payload.get("layer_density_scale", -1.0)) != float(args.layer_density_scale):
        return False, "layer_density_scale_mismatch"
    expected_calibration_json = run_dir.parent.parent / "eval" / "eval_test" / "aggregated" / "layer_density_calibration.json"
    payload_calibration_json = str(payload.get("layer_density_calibration_json", "")).strip()
    if expected_calibration_json.exists():
        if not same_path(payload_calibration_json, expected_calibration_json):
            return False, "layer_density_calibration_json_mismatch"
    if float(payload.get("layer_density_calibration_min_scale", -1.0)) != float(args.layer_density_calibration_min_scale):
        return False, "layer_density_calibration_min_scale_mismatch"
    if float(payload.get("layer_density_calibration_max_scale", -1.0)) != float(args.layer_density_calibration_max_scale):
        return False, "layer_density_calibration_max_scale_mismatch"
    if bool(payload.get("no_layer_constraint", False)) != bool(args.no_layer_constraint):
        return False, "no_layer_constraint_mismatch"
    if bool(payload.get("seismic_bound_extension_enabled", True)) != bool(not args.disable_seismic_bound_extension):
        return False, "seismic_bound_extension_flag_mismatch"
    if float(payload.get("layer_top_boundary_ms", float("nan"))) != float(args.layer_top_boundary_ms):
        return False, "layer_top_boundary_ms_mismatch"
    if float(payload.get("layer_bottom_boundary_ms", float("nan"))) != float(args.layer_bottom_boundary_ms):
        return False, "layer_bottom_boundary_ms_mismatch"
    if int(payload.get("window_size", -1)) != int(args.window_size):
        return False, "window_size_mismatch"
    if float(payload.get("z_step_ms", -1.0)) != float(args.z_step_ms):
        return False, "z_step_ms_mismatch"
    if float(payload.get("overlap_ratio", -1.0)) != float(args.overlap_ratio):
        return False, "overlap_ratio_mismatch"
    if int(payload.get("max_total_patches_per_window", -1)) != int(args.max_total_patches_per_window):
        return False, "max_total_patches_per_window_mismatch"
    if bool(payload.get("save_window_csv", False)) != bool(args.save_window_csv):
        return False, "save_window_csv_mismatch"
    if bool(payload.get("save_window_packages", False)) != bool(args.save_window_packages):
        return False, "save_window_packages_mismatch"
    expected_time_min = float(args.time_min) if args.time_min is not None else None
    expected_time_max = float(args.time_max) if args.time_max is not None else None
    if payload.get("time_min", None) != expected_time_min:
        return False, "time_min_mismatch"
    if payload.get("time_max", None) != expected_time_max:
        return False, "time_max_mismatch"

    selected_unit_ids = read_csv_unit_ids(selected_units_csv)
    if selected_unit_ids != expected_unit_ids:
        return False, f"selected_unit_ids_mismatch expected={len(expected_unit_ids)} actual={len(selected_unit_ids)}"
    if int(payload.get("selected_unit_count", -1)) != len(expected_unit_ids):
        return False, "selected_unit_count_mismatch"
    expected_output_paths: list[Path] = []
    for unit_id in expected_unit_ids:
        unit_dir = units_dir / unit_id
        for filename in DEFAULT_REQUIRED_UNIT_FILES:
            expected_output_paths.append(unit_dir / filename)
    missing_outputs = [str(path) for path in wait_for_existing_files(expected_output_paths)]
    if missing_outputs:
        preview = ", ".join(missing_outputs[:8])
        return False, f"missing_unit_outputs({len(missing_outputs)}): {preview}"
    return True, f"validated_units={len(expected_unit_ids)}"


def validate_smoke(paths: PipelinePaths, args: argparse.Namespace) -> tuple[bool, str]:
    expected_unit_ids = collect_rect_unit_ids(
        block_x_start=args.smoke_block_x_start,
        block_x_end=args.smoke_block_x_end,
        block_y_start=args.smoke_block_y_start,
        block_y_end=args.smoke_block_y_end,
        limit_units=args.smoke_limit_units,
    )
    return validate_production_run(
        run_dir=paths.smoke_run_dir,
        expected_unit_ids=expected_unit_ids,
        shard_index=0,
        shard_count=1,
        args=args,
        registry_py=paths.train_registry_py,
        expected_artifact_profile=str(args.smoke_artifact_profile),
    )


def full_shard_count(args: argparse.Namespace) -> int:
    if args.full_unit_shard_count is not None:
        return max(1, int(args.full_unit_shard_count))
    return max(1, len(args.gpu_ids))


def validate_full(paths: PipelinePaths, args: argparse.Namespace) -> tuple[bool, str]:
    base_unit_ids = collect_rect_unit_ids(
        block_x_start=args.full_block_x_start,
        block_x_end=args.full_block_x_end,
        block_y_start=args.full_block_y_start,
        block_y_end=args.full_block_y_end,
    )
    shard_count = full_shard_count(args)
    invalid_shards: list[str] = []
    for shard_index in range(shard_count):
        expected_unit_ids = shard_unit_ids(base_unit_ids, shard_index=shard_index, shard_count=shard_count)
        valid, detail = validate_production_run(
            run_dir=paths.full_run_dir,
            expected_unit_ids=expected_unit_ids,
            shard_index=shard_index,
            shard_count=shard_count,
            args=args,
            registry_py=paths.train_registry_py,
            expected_artifact_profile=str(args.artifact_profile),
        )
        if not valid:
            invalid_shards.append(f"{shard_index}:{detail}")
    if invalid_shards:
        return False, " | ".join(invalid_shards[:4])
    return True, f"validated_shards={shard_count}, total_units={len(base_unit_ids)}"


def validate_merge(paths: PipelinePaths, args: argparse.Namespace) -> tuple[bool, str]:
    summary_path = paths.merge_run_dir / "merge_summary.json"
    payload = read_json(summary_path)
    if payload is None:
        return False, f"missing_or_invalid_json: {summary_path}"
    if not same_path(payload.get("units_root", ""), paths.full_run_dir / "units"):
        return False, "units_root_mismatch"
    if str(payload.get("vtk_name", "")) != str(args.merge_vtk_name):
        return False, "vtk_name_mismatch"
    output_vtk = Path(str(payload.get("output_vtk", ""))) if str(payload.get("output_vtk", "")).strip() else None
    if output_vtk is None or not output_vtk.exists():
        return False, "missing_output_vtk"
    expected_unit_count = len(
        collect_rect_unit_ids(
            block_x_start=args.full_block_x_start,
            block_x_end=args.full_block_x_end,
            block_y_start=args.full_block_y_start,
            block_y_end=args.full_block_y_end,
        )
    )
    if int(payload.get("selected_unit_count", -1)) != expected_unit_count:
        return False, "selected_unit_count_mismatch"
    return True, f"merged_unit_count={expected_unit_count}"


def validate_postprocess(paths: PipelinePaths, args: argparse.Namespace) -> tuple[bool, str]:
    payload = read_json(paths.postprocess_summary_json)
    if payload is None:
        return False, f"missing_or_invalid_json: {paths.postprocess_summary_json}"
    if not same_path(payload.get("input_vtk", ""), paths.merge_output_vtk):
        return False, "input_vtk_mismatch"
    if not same_path(payload.get("output_vtk", ""), paths.postprocess_output_vtk):
        return False, "output_vtk_mismatch"
    if int(payload.get("phase", -1)) != int(args.postprocess_phase):
        return False, "postprocess_phase_mismatch"
    if bool(payload.get("boundary_connect", False)) != bool(args.postprocess_boundary_connect):
        return False, "postprocess_boundary_connect_mismatch"
    if bool(payload.get("bc_seam_fill", False)) != bool(args.postprocess_bc_seam_fill):
        return False, "postprocess_bc_seam_fill_mismatch"
    if not paths.postprocess_output_vtk.exists():
        return False, "missing_output_vtk"
    stats = payload.get("stats", {}) if isinstance(payload.get("stats"), dict) else {}
    output_count = int(stats.get("output_count", 0)) if stats else 0
    if output_count <= 0:
        return False, "postprocess_output_count<=0"
    return True, f"postprocess_output_count={output_count}"


def validate_fault_postfusion(paths: PipelinePaths, args: argparse.Namespace) -> tuple[bool, str]:
    payload = read_json(paths.fault_summary_json)
    if payload is None:
        return False, f"missing_or_invalid_json: {paths.fault_summary_json}"
    if not same_path(payload.get("input_vtk", ""), paths.postprocess_output_vtk):
        return False, "input_vtk_mismatch"
    if not same_path(payload.get("fault_patches_root", ""), Path(args.fault_patches_root)):
        return False, "fault_patches_root_mismatch"
    if float(payload.get("fault_half_band_ms", -1.0)) != float(args.fault_half_band_ms):
        return False, "fault_half_band_ms_mismatch"
    if float(payload.get("fault_remove_ms", -1.0)) != float(args.fault_remove_ms):
        return False, "fault_remove_ms_mismatch"
    if float(payload.get("fault_transition_ms", -1.0)) != float(args.fault_transition_ms):
        return False, "fault_transition_ms_mismatch"
    if float(payload.get("surface_max_fragment_area_ratio", -1.0)) != float(args.fault_surface_max_fragment_area_ratio):
        return False, "surface_max_fragment_area_ratio_mismatch"
    final_vtk_text = str(payload.get("final_vtk", "")).strip()
    if not final_vtk_text:
        return False, "missing_final_vtk"
    final_vtk = Path(final_vtk_text)
    if not final_vtk.exists():
        return False, "final_vtk_not_found"
    return True, f"final_vtk={final_vtk}"


def stage_log_path(paths: PipelinePaths, prefix: str) -> Path:
    return paths.logs_dir / prefix


def update_stage_state(
    state: dict[str, Any],
    state_path: Path,
    stage_name: str,
    **fields: Any,
) -> None:
    stage_payload = dict(state.get("stages", {}).get(stage_name, {}))
    stage_payload.update(fields)
    state.setdefault("stages", {})[stage_name] = stage_payload
    save_state(state_path, state)


def build_stats_args(paths: PipelinePaths, args: argparse.Namespace) -> list[str]:
    extra = [
        "--unit-dfn-root", str(Path(args.unit_dfn_root).resolve()),
        "--trace-header-csv", str(Path(args.trace_header_csv).resolve()),
        "--output-root", str(paths.stats_output_root.resolve()),
        "--run-name", "stats",
        "--docx-path", str(Path(args.docx_path).resolve()),
        "--xy-resolution", str(int(args.xy_resolution)),
        "--z-step-ms", str(float(args.z_step_ms)),
        "--window-sizes", str(int(args.window_size)),
        "--overlap-ratio", str(float(args.overlap_ratio)),
        "--candidate-k", *[str(int(v)) for v in args.candidate_k],
    ]
    if args.limit_units is not None:
        extra.extend(["--limit-units", str(int(args.limit_units))])
    return extra


def build_package_args(paths: PipelinePaths, args: argparse.Namespace) -> list[str]:
    extra = [
        "--unit-dfn-root", str(Path(args.unit_dfn_root).resolve()),
        "--trace-header-csv", str(Path(args.trace_header_csv).resolve()),
        "--sgy-file", str(Path(args.sgy_file).resolve()),
        "--stats-run-dir", str(paths.stats_run_dir.resolve()),
        "--output-root", str(paths.dataset_output_root.resolve()),
        "--run-name", "dataset",
        "--docx-path", str(Path(args.docx_path).resolve()),
        "--xy-resolution", str(int(args.xy_resolution)),
        "--z-step-ms", str(float(args.z_step_ms)),
        "--window-size", str(int(args.window_size)),
        "--overlap-ratio", str(float(args.overlap_ratio)),
        "--input-channels", *[str(v) for v in args.input_channels],
        "--input-dtype", str(args.input_dtype),
        "--geom-dtype", str(args.geom_dtype),
    ]
    if args.limit_units is not None:
        extra.extend(["--limit-units", str(int(args.limit_units))])
    if bool(args.omit_count_volume):
        extra.append("--omit-count-volume")
    return extra


def build_split_args(paths: PipelinePaths, args: argparse.Namespace) -> list[str]:
    extra = [
        "--dataset-run-dir", str(paths.dataset_run_dir.resolve()),
        "--output-root", str(paths.split_output_root.resolve()),
        "--run-name", "split",
        "--docx-path", str(Path(args.docx_path).resolve()),
        "--train-ratio", str(float(args.train_ratio)),
        "--val-ratio", str(float(args.val_ratio)),
        "--seed", str(int(args.split_seed)),
    ]
    if args.train_unit_count is not None:
        extra.extend(["--train-unit-count", str(int(args.train_unit_count))])
    if args.val_unit_count is not None:
        extra.extend(["--val-unit-count", str(int(args.val_unit_count))])
    if args.test_unit_count is not None:
        extra.extend(["--test-unit-count", str(int(args.test_unit_count))])
    if args.fixed_unit_split_csv is not None:
        extra.extend(["--fixed-unit-split-csv", str(Path(args.fixed_unit_split_csv).resolve())])
    return extra


def build_train_args(paths: PipelinePaths, args: argparse.Namespace, overwrite_existing: bool) -> list[str]:
    extra = [
        "--split-run-dir", str(paths.split_run_dir.resolve()),
        "--unit-dfn-root", str(Path(args.unit_dfn_root).resolve()),
        "--output-root", str(paths.train_output_root.resolve()),
        "--run-name-prefix", "layerwise_baseline",
        "--registry-output-py", str(paths.train_registry_py.resolve()),
        "--docx-path", str(Path(args.docx_path).resolve()),
        "--gpu-ids", *[str(int(v)) for v in args.gpu_ids],
        "--slots-per-voxel", str(int(args.slots_per_voxel)),
        "--base-channels", str(int(args.base_channels)),
        "--batch-size", str(int(args.batch_size)),
        "--num-workers", str(int(args.num_workers)),
        "--torch-num-threads", str(int(args.torch_num_threads)),
        "--torch-num-interop-threads", str(int(args.torch_num_interop_threads)),
        "--epochs", str(int(args.epochs)),
        "--learning-rate", str(float(args.learning_rate)),
        "--weight-decay", str(float(args.weight_decay)),
        "--center-positive-weight", str(float(args.center_positive_weight)),
        "--center-negative-weight", str(float(args.center_negative_weight)),
        "--center-focal-gamma", str(float(args.center_focal_gamma)),
        "--count-positive-weight", str(float(args.count_positive_weight)),
        "--count-negative-weight", str(float(args.count_negative_weight)),
        "--center-loss-weight", str(float(args.center_loss_weight)),
        "--count-loss-weight", str(float(args.count_loss_weight)),
        "--calibration-center-thresholds", *[str(float(v)) for v in args.calibration_center_thresholds],
        "--calibration-count-thresholds", *[str(float(v)) for v in args.calibration_count_thresholds],
        "--calibration-window-weight", str(float(args.calibration_window_weight)),
        "--seed", str(int(args.seed)),
    ]
    if args.train_limit_samples is not None:
        extra.extend(["--train-limit-samples", str(int(args.train_limit_samples))])
    if args.val_limit_samples is not None:
        extra.extend(["--val-limit-samples", str(int(args.val_limit_samples))])
    if args.max_train_steps is not None:
        extra.extend(["--max-train-steps", str(int(args.max_train_steps))])
    if args.max_val_steps is not None:
        extra.extend(["--max-val-steps", str(int(args.max_val_steps))])
    if bool(args.disable_amp):
        extra.append("--disable-amp")
    if bool(args.no_cache_raw_packages):
        extra.append("--no-cache-raw-packages")
    if bool(args.no_progress):
        extra.append("--no-progress")
    if bool(overwrite_existing):
        extra.append("--overwrite-existing")
    return extra


def build_eval_args(paths: PipelinePaths, args: argparse.Namespace) -> list[str]:
    extra = [
        "--split-run-dir", str(paths.split_run_dir.resolve()),
        "--layer-model-registry-py", str(paths.train_registry_py.resolve()),
        "--split-name", str(args.eval_split_name),
        "--output-root", str(paths.eval_output_root.resolve()),
        "--run-name", "eval_test",
        "--docx-path", str(Path(args.docx_path).resolve()),
        "--unit-dfn-root", str(Path(args.unit_dfn_root).resolve()),
        "--center-threshold", str(float(args.eval_center_threshold)),
        "--decode-mode", str(args.eval_decode_mode),
        "--relaxed-min-count", str(int(args.eval_relaxed_min_count)),
        "--count-activation-threshold", str(float(args.eval_count_activation_threshold)),
        "--min-count-if-active", str(int(args.eval_min_count_if_active)),
        "--max-slots-per-voxel", str(int(args.max_slots_per_voxel)),
        "--max-total-patches-per-window", str(int(args.max_total_patches_per_window)),
        "--layer-density-source", str(args.layer_density_source),
        "--layer-density-scale", str(float(args.layer_density_scale)),
        "--layer-density-calibration-min-scale", str(float(args.layer_density_calibration_min_scale)),
        "--layer-density-calibration-max-scale", str(float(args.layer_density_calibration_max_scale)),
        "--device", "cuda",
    ]
    if bool(args.disable_layer_density_control):
        extra.append("--disable-layer-density-control")
    if args.eval_limit_samples is not None:
        extra.extend(["--limit-samples", str(int(args.eval_limit_samples))])
    if args.eval_limit_units is not None:
        extra.extend(["--limit-units", str(int(args.eval_limit_units))])
    return extra


def build_production_common_args(
    output_root: Path,
    run_name: str,
    args: argparse.Namespace,
    registry_py: Path,
    artifact_profile: str,
) -> list[str]:
    calibration_json = output_root.parent / "eval" / "eval_test" / "aggregated" / "layer_density_calibration.json"
    extra = [
        "--layer-model-registry-py", str(registry_py.resolve()),
        "--unit-dfn-root", str(Path(args.unit_dfn_root).resolve()),
        "--surface-dir", str(Path(args.surface_dir).resolve()),
        "--trace-header-csv", str(Path(args.trace_header_csv).resolve()),
        "--sgy-file", str(Path(args.sgy_file).resolve()),
        "--output-root", str(output_root.resolve()),
        "--run-name", run_name,
        "--docx-path", str(Path(args.docx_path).resolve()),
        "--window-size", str(int(args.window_size)),
        "--z-step-ms", str(float(args.z_step_ms)),
        "--overlap-ratio", str(float(args.overlap_ratio)),
        "--layer-top-boundary-ms", str(float(args.layer_top_boundary_ms)),
        "--layer-bottom-boundary-ms", str(float(args.layer_bottom_boundary_ms)),
        "--input-channels", *[str(v) for v in args.input_channels],
        "--center-threshold", str(float(args.center_threshold)),
        "--patch-scale-factor", str(float(args.patch_scale_factor)),
        "--decode-mode", str(args.decode_mode),
        "--relaxed-min-count", str(int(args.relaxed_min_count)),
        "--count-activation-threshold", str(float(args.count_activation_threshold)),
        "--min-count-if-active", str(int(args.min_count_if_active)),
        "--max-slots-per-voxel", str(int(args.max_slots_per_voxel)),
        "--max-total-patches-per-window", str(int(args.max_total_patches_per_window)),
        "--layer-density-source", str(args.layer_density_source),
        "--layer-density-scale", str(float(args.layer_density_scale)),
        "--layer-density-calibration-min-scale", str(float(args.layer_density_calibration_min_scale)),
        "--layer-density-calibration-max-scale", str(float(args.layer_density_calibration_max_scale)),
        "--dedupe-xy-tol-m", str(float(args.dedupe_xy_tol_m)),
        "--dedupe-time-tol-ms", str(float(args.dedupe_time_tol_ms)),
        "--dedupe-azimuth-tol-deg", str(float(args.dedupe_azimuth_tol_deg)),
        "--dedupe-dip-tol-deg", str(float(args.dedupe_dip_tol_deg)),
        "--device", "cuda",
        "--artifact-profile", str(artifact_profile),
        "--display-z-scale", str(float(args.display_z_scale)),
        "--partial-save-every-windows", str(int(args.partial_save_every_windows)),
    ]
    if calibration_json.exists():
        extra.extend(["--layer-density-calibration-json", str(calibration_json.resolve())])
    if bool(args.no_progress):
        extra.append("--no-progress")
    if bool(args.vtk_no_invert_time):
        extra.append("--vtk-no-invert-time")
    if bool(args.save_window_csv):
        extra.append("--save-window-csv")
    if bool(args.save_window_packages):
        extra.append("--save-window-packages")
    if bool(args.no_layer_constraint):
        extra.append("--no-layer-constraint")
    if bool(args.disable_seismic_bound_extension):
        extra.append("--disable-seismic-bound-extension")
    if bool(args.disable_layer_density_control):
        extra.append("--disable-layer-density-control")
    if args.time_min is not None:
        extra.extend(["--time-min", str(float(args.time_min))])
    if args.time_max is not None:
        extra.extend(["--time-max", str(float(args.time_max))])
    return extra


def build_smoke_args(paths: PipelinePaths, args: argparse.Namespace) -> list[str]:
    extra = build_production_common_args(
        output_root=paths.smoke_output_root,
        run_name="smoke_infer",
        args=args,
        registry_py=paths.train_registry_py,
        artifact_profile=str(args.smoke_artifact_profile),
    )
    extra.extend(
        [
            "--block-x-start", str(int(args.smoke_block_x_start)),
            "--block-x-end", str(int(args.smoke_block_x_end)),
            "--block-y-start", str(int(args.smoke_block_y_start)),
            "--block-y-end", str(int(args.smoke_block_y_end)),
        ]
    )
    if args.smoke_limit_units is not None:
        extra.extend(["--limit-units", str(int(args.smoke_limit_units))])
    return extra


def build_full_shard_args(paths: PipelinePaths, args: argparse.Namespace, shard_index: int, shard_count: int) -> list[str]:
    extra = build_production_common_args(
        output_root=paths.full_output_root,
        run_name="full_infer",
        args=args,
        registry_py=paths.train_registry_py,
        artifact_profile=str(args.artifact_profile),
    )
    extra.extend(
        [
            "--block-x-start", str(int(args.full_block_x_start)),
            "--block-x-end", str(int(args.full_block_x_end)),
            "--block-y-start", str(int(args.full_block_y_start)),
            "--block-y-end", str(int(args.full_block_y_end)),
            "--unit-shard-index", str(int(shard_index)),
            "--unit-shard-count", str(int(shard_count)),
        ]
    )
    return extra


def build_merge_args(paths: PipelinePaths, args: argparse.Namespace) -> list[str]:
    return [
        "--units-root", str((paths.full_run_dir / "units").resolve()),
        "--block-x-start", str(int(args.full_block_x_start)),
        "--block-x-end", str(int(args.full_block_x_end)),
        "--block-y-start", str(int(args.full_block_y_start)),
        "--block-y-end", str(int(args.full_block_y_end)),
        "--vtk-name", str(args.merge_vtk_name),
        "--output-root", str(paths.merge_output_root.resolve()),
        "--run-name", "merge_full",
        "--docx-path", str(Path(args.docx_path).resolve()),
        "--output-vtk-name", str(paths.merge_output_vtk.name),
    ]


def build_postprocess_args(paths: PipelinePaths, args: argparse.Namespace) -> list[str]:
    extra = [
        "--input-vtk", str(paths.merge_output_vtk.resolve()),
        "--output-root", str(paths.postprocess_output_root.resolve()),
        "--run-name", "postprocess_full",
        "--output-vtk-name", str(paths.postprocess_output_vtk.name),
        "--phase", str(int(args.postprocess_phase)),
        "--compute-backend", str(args.postprocess_compute_backend),
        "--max-cpu-threads", str(int(args.postprocess_max_cpu_threads)),
        "--gpu-tile-points", str(int(args.postprocess_gpu_tile_points)),
        "--docx-path", str(Path(args.docx_path).resolve()),
        "--overwrite",
    ]
    if bool(args.postprocess_boundary_connect):
        extra.append("--boundary-connect")
    if bool(args.postprocess_bc_seam_fill):
        extra.append("--bc-seam-fill")
    return extra


def build_fault_postfusion_args(paths: PipelinePaths, args: argparse.Namespace) -> list[str]:
    return [
        "--input-vtk", str(paths.postprocess_output_vtk.resolve()),
        "--fault-patches-root", str(Path(args.fault_patches_root).resolve()),
        "--block-x-start", str(int(args.full_block_x_start)),
        "--block-x-end", str(int(args.full_block_x_end)),
        "--block-y-start", str(int(args.full_block_y_start)),
        "--block-y-end", str(int(args.full_block_y_end)),
        "--output-root", str(paths.fault_output_root.resolve()),
        "--run-name", "fault_postfusion_full",
        "--fault-half-band-ms", str(float(args.fault_half_band_ms)),
        "--fault-remove-ms", str(float(args.fault_remove_ms)),
        "--fault-transition-ms", str(float(args.fault_transition_ms)),
        "--surface-max-fragment-area-ratio", str(float(args.fault_surface_max_fragment_area_ratio)),
        "--docx-path", str(Path(args.docx_path).resolve()),
        "--overwrite",
    ]


def run_simple_stage(
    *,
    stage_name: str,
    script_path: Path,
    script_args: list[str],
    validator: Any,
    paths: PipelinePaths,
    args: argparse.Namespace,
    state: dict[str, Any],
    force_run: bool,
    log_name: str,
    env_overrides: dict[str, str] | None = None,
) -> StageResult:
    log_path = stage_log_path(paths, log_name)
    if not force_run:
        valid, detail = validator(paths, args)
        if valid:
            update_stage_state(
                state,
                paths.state_path,
                stage_name,
                status="validated_existing",
                finished_at=now_text(),
                detail=detail,
                log_path=str(log_path),
            )
            print(f"[{stage_name}] skip: {detail}")
            return StageResult(executed=False, detail=detail)

    max_attempts = max(1, int(args.max_stage_retries) + 1)
    last_detail = ""
    for attempt in range(1, max_attempts + 1):
        cmd = build_conda_python_cmd(args.conda_env, script_path, script_args)
        update_stage_state(
            state,
            paths.state_path,
            stage_name,
            status="running",
            started_at=now_text(),
            attempt=attempt,
            command=quote_cmd(cmd),
            log_path=str(log_path),
        )
        print(f"[{stage_name}] attempt {attempt}/{max_attempts}")
        return_code = run_command(cmd=cmd, log_path=log_path, cwd=THIS_DIR, env_overrides=env_overrides)
        valid, detail = validator(paths, args)
        last_detail = detail
        if return_code == 0 and valid:
            update_stage_state(
                state,
                paths.state_path,
                stage_name,
                status="completed",
                finished_at=now_text(),
                detail=detail,
                return_code=return_code,
            )
            print(f"[{stage_name}] done: {detail}")
            return StageResult(executed=True, detail=detail)
        update_stage_state(
            state,
            paths.state_path,
            stage_name,
            status="failed",
            finished_at=now_text(),
            detail=detail,
            return_code=return_code,
        )
        print(f"[{stage_name}] failed: return_code={return_code}, detail={detail}")
        if attempt < max_attempts:
            time.sleep(max(0.0, float(args.retry_sleep_seconds)))
    raise RuntimeError(f"stage={stage_name} failed after {max_attempts} attempts: {last_detail}")


def run_full_infer_stage(
    *,
    paths: PipelinePaths,
    args: argparse.Namespace,
    state: dict[str, Any],
    force_run: bool,
) -> StageResult:
    stage_name = "full_infer"
    shard_count = full_shard_count(args)
    if shard_count <= 0:
        raise ValueError("full_unit_shard_count must be >= 1")
    base_unit_ids = collect_rect_unit_ids(
        block_x_start=args.full_block_x_start,
        block_x_end=args.full_block_x_end,
        block_y_start=args.full_block_y_start,
        block_y_end=args.full_block_y_end,
    )

    invalid_shards: list[int] = []
    if not force_run:
        valid, detail = validate_full(paths, args)
        if valid:
            update_stage_state(
                state,
                paths.state_path,
                stage_name,
                status="validated_existing",
                finished_at=now_text(),
                detail=detail,
                log_dir=str(paths.logs_dir),
            )
            print(f"[{stage_name}] skip: {detail}")
            return StageResult(executed=False, detail=detail)
        for shard_index in range(shard_count):
            expected_unit_ids = shard_unit_ids(base_unit_ids, shard_index=shard_index, shard_count=shard_count)
            shard_valid, _ = validate_production_run(
                run_dir=paths.full_run_dir,
                expected_unit_ids=expected_unit_ids,
                shard_index=shard_index,
                shard_count=shard_count,
                args=args,
                registry_py=paths.train_registry_py,
                expected_artifact_profile=str(args.artifact_profile),
            )
            if not shard_valid:
                invalid_shards.append(shard_index)
    else:
        invalid_shards = list(range(shard_count))

    max_attempts = max(1, int(args.max_shard_retries) + 1)
    last_detail = ""
    for attempt in range(1, max_attempts + 1):
        if not invalid_shards:
            valid, detail = validate_full(paths, args)
            if valid:
                update_stage_state(
                    state,
                    paths.state_path,
                    stage_name,
                    status="completed",
                    finished_at=now_text(),
                    detail=detail,
                    shard_count=shard_count,
                )
                print(f"[{stage_name}] done: {detail}")
                return StageResult(executed=bool(force_run), detail=detail)
            last_detail = detail
        update_stage_state(
            state,
            paths.state_path,
            stage_name,
            status="running",
            started_at=now_text(),
            attempt=attempt,
            shard_count=shard_count,
            rerun_shards=list(invalid_shards),
        )
        print(f"[{stage_name}] attempt {attempt}/{max_attempts}, shards={invalid_shards}")
        processes: list[dict[str, Any]] = []
        for local_idx, shard_index in enumerate(invalid_shards):
            shard_suffix = build_shard_suffix(shard_index=shard_index, shard_count=shard_count)
            log_path = stage_log_path(paths, f"07_full_infer{shard_suffix}.log")
            gpu_id = args.gpu_ids[local_idx % len(args.gpu_ids)]
            shard_args = build_full_shard_args(paths, args, shard_index=shard_index, shard_count=shard_count)
            cmd = build_conda_python_cmd(args.conda_env, PRODUCTION_SCRIPT, shard_args)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fp = log_path.open("a", encoding="utf-8")
            log_fp.write(f"\n[{now_text()}] COMMAND START\n")
            log_fp.write(quote_cmd(cmd) + "\n\n")
            log_fp.flush()
            process = subprocess.Popen(
                cmd,
                cwd=str(THIS_DIR),
                env=env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
            )
            processes.append(
                {
                    "shard_index": shard_index,
                    "gpu_id": gpu_id,
                    "cmd": cmd,
                    "log_path": log_path,
                    "log_fp": log_fp,
                    "process": process,
                }
            )

        failed_by_return_code: list[int] = []
        for proc_info in processes:
            return_code = int(proc_info["process"].wait())
            proc_info["log_fp"].write(f"\n[{now_text()}] COMMAND END return_code={return_code}\n")
            proc_info["log_fp"].flush()
            proc_info["log_fp"].close()
            if return_code != 0:
                failed_by_return_code.append(int(proc_info["shard_index"]))

        invalid_next: list[int] = []
        for shard_index in range(shard_count):
            expected_unit_ids = shard_unit_ids(base_unit_ids, shard_index=shard_index, shard_count=shard_count)
            shard_valid, detail = validate_production_run(
                run_dir=paths.full_run_dir,
                expected_unit_ids=expected_unit_ids,
                shard_index=shard_index,
                shard_count=shard_count,
                args=args,
                registry_py=paths.train_registry_py,
                expected_artifact_profile=str(args.artifact_profile),
            )
            if not shard_valid:
                invalid_next.append(shard_index)
                last_detail = detail

        if not invalid_next:
            valid, detail = validate_full(paths, args)
            if valid:
                update_stage_state(
                    state,
                    paths.state_path,
                    stage_name,
                    status="completed",
                    finished_at=now_text(),
                    detail=detail,
                    shard_count=shard_count,
                )
                print(f"[{stage_name}] done: {detail}")
                return StageResult(executed=True, detail=detail)
            last_detail = detail

        invalid_shards = sorted(set(invalid_next + failed_by_return_code))
        update_stage_state(
            state,
            paths.state_path,
            stage_name,
            status="failed",
            finished_at=now_text(),
            detail=last_detail,
            failed_shards=list(invalid_shards),
        )
        print(f"[{stage_name}] failed shards: {invalid_shards}, detail={last_detail}")
        if attempt < max_attempts:
            time.sleep(max(0.0, float(args.retry_sleep_seconds)))
    raise RuntimeError(f"stage={stage_name} failed after {max_attempts} attempts: {last_detail}")


def requested_stages(args: argparse.Namespace) -> list[str]:
    stage_order = [stage for stage in STAGE_ORDER if not (stage == "smoke_infer" and bool(args.skip_smoke_infer))]
    if not args.stop_after:
        return stage_order
    if str(args.stop_after) not in stage_order:
        raise ValueError(f"stop_after stage is disabled by current flags: {args.stop_after}")
    stop_index = stage_order.index(str(args.stop_after))
    return stage_order[: stop_index + 1]


def validate_initial_conditions(args: argparse.Namespace, requested_stage_names: list[str]) -> None:
    ensure_scripts_exist()
    if not Path(args.unit_dfn_root).exists():
        raise FileNotFoundError(f"unit_dfn_root not found: {args.unit_dfn_root}")
    if not Path(args.trace_header_csv).exists():
        raise FileNotFoundError(f"trace_header_csv not found: {args.trace_header_csv}")
    if not Path(args.sgy_file).exists():
        raise FileNotFoundError(f"sgy_file not found: {args.sgy_file}")
    if not Path(args.surface_dir).exists():
        raise FileNotFoundError(f"surface_dir not found: {args.surface_dir}")
    if args.fixed_unit_split_csv is not None and not Path(args.fixed_unit_split_csv).exists():
        raise FileNotFoundError(f"fixed_unit_split_csv not found: {args.fixed_unit_split_csv}")
    if args.time_min is None and args.time_max is not None:
        raise ValueError("time_min and time_max must be provided together")
    if args.time_min is not None and args.time_max is None:
        raise ValueError("time_min and time_max must be provided together")
    if not args.gpu_ids:
        raise ValueError("gpu_ids must not be empty")
    if float(args.layer_bottom_boundary_ms) <= float(args.layer_top_boundary_ms):
        raise ValueError("layer_bottom_boundary_ms must be greater than layer_top_boundary_ms")
    if "fault_postfusion" in requested_stage_names and not Path(args.fault_patches_root).exists():
        raise FileNotFoundError(f"fault_patches_root not found: {args.fault_patches_root}")


def apply_restart_from_stage(args: argparse.Namespace) -> None:
    restart_stage = str(getattr(args, "restart_from_stage", "") or "").strip()
    if not restart_stage:
        return
    args.resume = True
    force_stages = [str(stage).strip() for stage in (args.force_stage or []) if str(stage).strip()]
    if restart_stage not in force_stages:
        force_stages.append(restart_stage)
    args.force_stage = force_stages


def main() -> None:
    args = build_parser().parse_args()
    apply_restart_from_stage(args)
    active_stages = requested_stages(args)
    validate_initial_conditions(args, active_stages)
    paths = build_paths(args)
    paths.pipeline_root.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)

    if paths.state_path.exists() and not bool(args.resume):
        raise FileExistsError(
            f"pipeline state already exists: {paths.state_path}. "
            "If you want to continue this run, re-run with --resume."
        )

    state = load_state(paths.state_path, run_name=str(args.run_name), args=args)
    save_state(paths.state_path, state)

    print(f"run_name: {args.run_name}")
    print(f"pipeline_root: {paths.pipeline_root}")
    print(f"state_path: {paths.state_path}")
    print(f"requested_stages: {active_stages}")

    executed_stages: set[str] = set()
    force_set = {str(stage) for stage in args.force_stage}

    stage_to_runner = {
        "stats": lambda force_run: run_simple_stage(
            stage_name="stats",
            script_path=STATS_SCRIPT,
            script_args=build_stats_args(paths, args),
            validator=validate_stats,
            paths=paths,
            args=args,
            state=state,
            force_run=force_run,
            log_name="01_stats.log",
        ),
        "package": lambda force_run: run_simple_stage(
            stage_name="package",
            script_path=PACKAGE_SCRIPT,
            script_args=build_package_args(paths, args),
            validator=validate_package,
            paths=paths,
            args=args,
            state=state,
            force_run=force_run,
            log_name="02_package.log",
        ),
        "split": lambda force_run: run_simple_stage(
            stage_name="split",
            script_path=SPLIT_SCRIPT,
            script_args=build_split_args(paths, args),
            validator=validate_split,
            paths=paths,
            args=args,
            state=state,
            force_run=force_run,
            log_name="03_split.log",
        ),
        "train_layerwise": lambda force_run: run_simple_stage(
            stage_name="train_layerwise",
            script_path=TRAIN_SCRIPT,
            script_args=build_train_args(
                paths,
                args,
                overwrite_existing=bool(force_run and args.retrain_layerwise),
            ),
            validator=validate_train,
            paths=paths,
            args=args,
            state=state,
            force_run=force_run,
            log_name="04_train_layerwise.log",
        ),
        "eval": lambda force_run: run_simple_stage(
            stage_name="eval",
            script_path=EVAL_SCRIPT,
            script_args=build_eval_args(paths, args),
            validator=validate_eval,
            paths=paths,
            args=args,
            state=state,
            force_run=force_run,
            log_name="05_eval.log",
            env_overrides={"CUDA_VISIBLE_DEVICES": str(int(args.gpu_ids[0]))},
        ),
        "smoke_infer": lambda force_run: run_simple_stage(
            stage_name="smoke_infer",
            script_path=PRODUCTION_SCRIPT,
            script_args=build_smoke_args(paths, args),
            validator=validate_smoke,
            paths=paths,
            args=args,
            state=state,
            force_run=force_run,
            log_name="06_smoke_infer.log",
            env_overrides={"CUDA_VISIBLE_DEVICES": str(int(args.gpu_ids[0]))},
        ),
        "full_infer": lambda force_run: run_full_infer_stage(
            paths=paths,
            args=args,
            state=state,
            force_run=force_run,
        ),
        "merge": lambda force_run: run_simple_stage(
            stage_name="merge",
            script_path=MERGE_SCRIPT,
            script_args=build_merge_args(paths, args),
            validator=validate_merge,
            paths=paths,
            args=args,
            state=state,
            force_run=force_run,
            log_name="08_merge.log",
        ),
        "postprocess": lambda force_run: run_simple_stage(
            stage_name="postprocess",
            script_path=POSTPROCESS_SCRIPT,
            script_args=build_postprocess_args(paths, args),
            validator=validate_postprocess,
            paths=paths,
            args=args,
            state=state,
            force_run=force_run,
            log_name="09_postprocess.log",
        ),
        "fault_postfusion": lambda force_run: run_simple_stage(
            stage_name="fault_postfusion",
            script_path=FAULT_POSTFUSION_SCRIPT,
            script_args=build_fault_postfusion_args(paths, args),
            validator=validate_fault_postfusion,
            paths=paths,
            args=args,
            state=state,
            force_run=force_run,
            log_name="10_fault_postfusion.log",
        ),
    }

    for stage_name in active_stages:
        dependency_executed = any(dep in executed_stages for dep in STAGE_DEPENDENCIES.get(stage_name, []))
        force_run = bool(stage_name in force_set or dependency_executed)
        result = stage_to_runner[stage_name](force_run)
        if result.executed:
            executed_stages.add(stage_name)

    summary = {
        "run_name": str(args.run_name),
        "pipeline_root": str(paths.pipeline_root),
        "state_path": str(paths.state_path),
        "train_registry_py": str(paths.train_registry_py),
        "smoke_run_dir": str(paths.smoke_run_dir),
        "full_run_dir": str(paths.full_run_dir),
        "merge_run_dir": str(paths.merge_run_dir),
        "merge_output_vtk": str(paths.merge_output_vtk),
        "postprocess_run_dir": str(paths.postprocess_run_dir),
        "postprocess_output_vtk": str(paths.postprocess_output_vtk),
        "fault_run_dir": str(paths.fault_run_dir),
        "fault_summary_json": str(paths.fault_summary_json),
        "completed_at": now_text(),
    }
    write_json(paths.pipeline_root / "pipeline_summary.json", summary)
    print(f"pipeline_summary: {paths.pipeline_root / 'pipeline_summary.json'}")
    print(f"train_registry_py: {paths.train_registry_py}")
    print(f"full_run_dir: {paths.full_run_dir}")
    print(f"merge_run_dir: {paths.merge_run_dir}")
    print(f"postprocess_run_dir: {paths.postprocess_run_dir}")
    print(f"fault_run_dir: {paths.fault_run_dir}")


if __name__ == "__main__":
    main()
