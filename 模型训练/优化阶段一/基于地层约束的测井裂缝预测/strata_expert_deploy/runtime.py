from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from workflow_paths import NEAREST_SCRIPT_PATH, RAW_POINT_SCRIPT_PATH, WORKFLOW_ROOT

PROJECT_ROOT = WORKFLOW_ROOT
OUTER_FLOW_SCRIPT = PROJECT_ROOT / "run_outer_holdout_expert_validation.py"

FLOW_RESULT_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果"
)
DEFAULT_SAVE_ROOT = FLOW_RESULT_ROOT / "deploy_runs"
DEFAULT_LOG_FEATURES = ["AC", "GR"]
MODULE_CACHE: dict[str, object] = {}


def sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(text)).strip("_")


def parse_json_list(raw: str) -> list[str]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    except Exception:
        pass
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    return [
        item.strip().strip("'\"")
        for item in raw.split(",")
        if item.strip().strip("'\"")
    ]


def load_module(module_path: Path, module_name: str):
    cache_key = f"{module_name}:{module_path}"
    if cache_key in MODULE_CACHE:
        return MODULE_CACHE[cache_key]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    added_to_syspath = False
    parent_dir = str(module_path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        added_to_syspath = True
    try:
        spec.loader.exec_module(module)
    finally:
        if added_to_syspath:
            try:
                sys.path.remove(parent_dir)
            except ValueError:
                pass
    MODULE_CACHE[cache_key] = module
    return module


def resolve_depth_column(df: pd.DataFrame) -> str:
    for col in ["TVD", "DEPT", "MD"]:
        if col in df.columns:
            return col
    raise ValueError("No depth column found. Expected one of TVD/DEPT/MD.")


def load_target_sample_df(
    target_sample_csv: Path,
    target_well_name: str,
) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(target_sample_csv, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Target sample CSV is empty: {target_sample_csv}")

    depth_col = resolve_depth_column(df)
    out_df = df.copy()
    out_df[depth_col] = pd.to_numeric(out_df[depth_col], errors="coerce")
    out_df = out_df[out_df[depth_col].notna()].copy()
    if "ROW_IN_WELL" in out_df.columns:
        out_df["ROW_IN_WELL"] = pd.to_numeric(out_df["ROW_IN_WELL"], errors="coerce")
        out_df = out_df.sort_values(["ROW_IN_WELL", depth_col], na_position="last").reset_index(drop=True)
    else:
        out_df = out_df.sort_values(depth_col).reset_index(drop=True)
        out_df["ROW_IN_WELL"] = np.arange(len(out_df), dtype=np.int64)
    out_df["WellName"] = str(target_well_name).strip()
    if "DepthForStrata" not in out_df.columns:
        out_df["DepthForStrata"] = out_df[depth_col]
    return out_df, depth_col


def sort_by_depth_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["TVD", "DEPT", "MD", "Depth", "DepthForStrata"]:
        if col in df.columns:
            out = df.copy()
            out[col] = pd.to_numeric(out[col], errors="coerce")
            return out.sort_values(col, na_position="last").reset_index(drop=True)
    return df.reset_index(drop=True)
