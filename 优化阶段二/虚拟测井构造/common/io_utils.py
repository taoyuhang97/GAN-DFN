from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read CSV: {path}") from last_error


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv_utf8(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")
