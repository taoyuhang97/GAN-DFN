# -*- coding: utf-8 -*-
"""Per-trace attribute-data bottom scan (T7-below extension boundary).

For every trace inside the configured target_block, find the deepest valid
sample of each of the three attribute volumes (Coherence / AntTrack /
CurvatureMax), where valid means finite and above ``attribute_null_abs_limit``
(sentinels such as -9999999 / -4700000 are excluded). The per-trace extension
bottom is the minimum of the three bottoms, capped by
``extension_max_bottom_ms``.

The result is used to build the extended horizon contract (T7' = bottom) and
to append the base-density extension band. The script is region-agnostic: the
same code runs for the 5 km demo block and for the full mine by changing only
``target_block`` in the config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio


CURRENT_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = CURRENT_DIR.parent
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))


ATTRIBUTE_KEYS = ("Coherence", "AntTrack", "CurvatureMax")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan per-trace attribute data bottoms for the T7-below extension.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_block_traces(config: dict[str, Any]) -> pd.DataFrame:
    header = pd.read_csv(config["trace_header_csv"], usecols=["TraceIdx", "X", "Y"], encoding="utf-8-sig")
    block = config["target_block"]
    mask = (
        header["X"].between(float(block["x_min"]), float(block["x_max"]))
        & header["Y"].between(float(block["y_min"]), float(block["y_max"]))
    )
    selected = header.loc[mask, ["TraceIdx", "X", "Y"]].copy()
    selected = selected.sort_values("TraceIdx").reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("target_block contains no trace-header rows")
    return selected


def read_trace_runs(handle: segyio.SegyFile, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    boundaries = np.where(np.diff(values) != 1)[0] + 1
    chunks = np.split(values, boundaries)
    parts: list[np.ndarray] = []
    for chunk in chunks:
        start = int(chunk[0])
        stop = int(chunk[-1]) + 1
        parts.append(np.asarray(handle.trace.raw[start:stop], dtype=np.float64))
    return np.concatenate(parts, axis=0)


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    output_root = Path(config["output_root"]).resolve()
    scan_dir = output_root / "attribute_bottom"
    scan_dir.mkdir(parents=True, exist_ok=True)

    selected = select_block_traces(config)
    trace_idx = selected["TraceIdx"].to_numpy(dtype=np.int64)
    table = np.load(config["horizon_source_table"], mmap_mode="r")
    lookup = {int(value): i for i, value in enumerate(np.asarray(table["TraceIdx"], dtype=np.int64))}
    rows = np.asarray([lookup[int(value)] for value in trace_idx], dtype=np.int64)
    t6 = np.asarray(table["T6"], dtype=np.float64)[rows]
    t7 = np.asarray(table["T7"], dtype=np.float64)[rows]
    shasan = np.asarray(table["ShasanPresent"], dtype=bool)[rows]
    shasi = np.asarray(table["ShasiPresent"], dtype=bool)[rows]

    null_limit = float(config.get("attribute_null_abs_limit", 1.0e6))
    max_bottom = float(config.get("extension_max_bottom_ms", 3674.0))
    volumes = {key: Path(config["volume_paths"][key]).resolve() for key in ATTRIBUTE_KEYS}
    for key, path in volumes.items():
        if not path.exists():
            raise FileNotFoundError(f"{key} volume not found: {path}")

    bottoms: dict[str, np.ndarray] = {}
    sample_axis: dict[str, np.ndarray] = {}
    for key, path in volumes.items():
        with segyio.open(str(path), "r", ignore_geometry=True) as handle:
            matrix = read_trace_runs(handle, trace_idx)
            samples = np.asarray(handle.samples, dtype=np.float64)
        sample_axis[key] = samples
        valid = np.isfinite(matrix) & (matrix > -null_limit)
        deepest = np.full(len(trace_idx), np.nan, dtype=np.float64)
        for i in range(len(trace_idx)):
            indices = np.flatnonzero(valid[i])
            if indices.size:
                deepest[i] = float(samples[int(indices[-1])])
        bottoms[key] = deepest
        print(f"[scan-bottom] {key}: traces={len(trace_idx)} with_data={int(np.isfinite(deepest).sum())}", flush=True)

    bottom_all = np.column_stack([bottoms[key] for key in ATTRIBUTE_KEYS])
    bottom = np.nanmin(bottom_all, axis=1)
    bottom = np.minimum(bottom, max_bottom)
    # Extension applies only on traces that originally have Shasi and whose
    # data bottom is strictly below the original T7.
    extension_ms = np.where(shasi & np.isfinite(bottom) & (bottom > t7), bottom - t7, 0.0)

    np.savez_compressed(
        scan_dir / "attribute_bottom.npz",
        TraceIdx=trace_idx.astype(np.int32),
        X=selected["X"].to_numpy(dtype=np.float64),
        Y=selected["Y"].to_numpy(dtype=np.float64),
        T6=t6.astype(np.float32),
        T7Original=t7.astype(np.float32),
        ShasanPresent=shasan.astype(np.uint8),
        ShasiPresent=shasi.astype(np.uint8),
        BottomMs=bottom.astype(np.float32),
        ExtensionMs=extension_ms.astype(np.float32),
    )

    ext_traces = int(np.count_nonzero(extension_ms > 0.0))
    stats = {
        "status": "pass",
        "config_path": str(args.config.resolve()),
        "target_block": config["target_block"],
        "trace_count": int(len(trace_idx)),
        "shasi_trace_count": int(shasi.sum()),
        "shasan_trace_count": int(shasan.sum()),
        "per_volume_deepest_valid": {key: int(np.isfinite(bottoms[key]).sum()) for key in ATTRIBUTE_KEYS},
        "extension_trace_count": ext_traces,
        "extension_trace_fraction": float(ext_traces / max(len(trace_idx), 1)),
        "extension_ms": {
            "min": float(extension_ms[extension_ms > 0].min()) if ext_traces else None,
            "median": float(np.median(extension_ms[extension_ms > 0])) if ext_traces else None,
            "p90": float(np.percentile(extension_ms[extension_ms > 0], 90)) if ext_traces else None,
            "max": float(extension_ms.max()),
        },
        "bottom_capped_by_max": int(np.count_nonzero((bottom > max_bottom - 1.0e-6) & (t7 < max_bottom - 1.0e-6))),
        "outputs": {
            "bottom_npz": str(scan_dir / "attribute_bottom.npz"),
        },
    }
    write_json(scan_dir / "attribute_bottom_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
