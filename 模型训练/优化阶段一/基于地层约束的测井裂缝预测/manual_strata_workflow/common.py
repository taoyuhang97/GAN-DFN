from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DEFAULT_WELL_FILES = {
    "车151HF": "车151HF_sample.csv",
    "车660-1": "车660-1_sample.csv",
    "车660-2": "车660-2_sample.csv",
    "车662": "车662_sample.csv",
    "车663": "车663_sample.csv",
    "车页1导眼": "车页1导眼_sample.csv",
}
WELL_NAME_ALIASES = {
    "车页1HF": "车页1导眼",
    "车页1导眼": "车页1导眼",
}
MANUAL_STRATA_CONFIG = {
    "车151HF": [
        {"name": "沙三段", "top": 3655.0, "base": 4935.0},
    ],
    "车660-1": [
        {"name": "沙三段", "top": 3927.0, "base": 4304.0},
    ],
    "车660-2": [
        {"name": "沙三段", "top": 4299.0, "base": 4321.0},
        {"name": "沙四段", "top": 4321.0, "base": 4752.0},
    ],
    "车662": [
        {"name": "沙三段", "top": 3475.0, "base": 3850.0},
        {"name": "沙四段", "top": 3850.0, "base": 3973.0},
    ],
    "车663": [
        {"name": "沙三段", "top": 3879.0, "base": 4220.0},
        {"name": "沙四段", "top": 4220.0, "base": 4281.0},
    ],
    "车页1导眼": [
        {"name": "沙三段", "top": 3500.0, "base": 3734.0},
        {"name": "沙四段", "top": 3734.0, "base": 3750.0},
    ],
}
DEFAULT_WELL_NAMES = list(DEFAULT_WELL_FILES.keys())
DEFAULT_TARGET_STRATA = ["沙三段", "沙四段"]


def sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(text)).strip("_")


def parse_json_list(raw: str) -> list[str]:
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError
        return [str(item).strip() for item in value if str(item).strip()]
    except Exception:
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1].strip()
        return [
            item.strip().strip("'\"")
            for item in raw.split(",")
            if item.strip().strip("'\"")
        ]


def parse_json_object(raw: str) -> dict:
    raw = str(raw).strip()
    if not raw:
        return {}
    maybe_path = Path(raw)
    if maybe_path.exists():
        raw = maybe_path.read_text(encoding="utf-8-sig")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def canonicalize_well_name(well_name: str) -> str:
    well_name = str(well_name).strip()
    return WELL_NAME_ALIASES.get(well_name, well_name)


def resolve_depth_column(df: pd.DataFrame) -> str:
    for col in ["TVD", "DEPT", "MD"]:
        if col in df.columns:
            return col
    raise ValueError("No depth column found. Expected one of TVD/DEPT/MD.")
