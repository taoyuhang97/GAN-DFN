#!/usr/bin/env python3
"""Replace a local 200 m DFN box with conditioned local patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replace large-profile local DFN patches")
    p.add_argument("--large-patches-csv", type=Path, required=True)
    p.add_argument("--local-patches-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--time-min-ms", type=float, required=True)
    p.add_argument("--time-max-ms", type=float, required=True)
    p.add_argument("--x-min", type=float, required=True)
    p.add_argument("--x-max", type=float, required=True)
    p.add_argument("--y-min", type=float, required=True)
    p.add_argument("--y-max", type=float, required=True)
    p.add_argument("--chunksize", type=int, default=50000)
    return p


def local_mask(frame: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    return (
        frame["CenterX"].between(float(args.x_min), float(args.x_max))
        & frame["CenterY"].between(float(args.y_min), float(args.y_max))
        & frame["CenterTime"].between(
            float(args.time_min_ms),
            float(args.time_max_ms),
        )
    )


def main() -> int:
    args = parser().parse_args()
    large_csv = args.large_patches_csv.resolve()
    local_csv = args.local_patches_csv.resolve()
    output_csv = args.output_csv.resolve()
    summary_json = args.summary_json.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")

    header = pd.read_csv(large_csv, nrows=0)
    output_columns = [*header.columns, "Local200Replacement"]
    local = pd.read_csv(local_csv, low_memory=False)
    required = {"FractureScale", "CenterX", "CenterY", "CenterTime"}
    missing = sorted(required - set(local.columns))
    if missing:
        raise ValueError(f"local patch CSV missing required columns: {missing}")

    first_write = True
    removed_rows = 0
    kept_rows = 0
    removed_counts: dict[str, int] = {}
    chunk_index = 0
    for chunk in pd.read_csv(
        large_csv,
        chunksize=int(args.chunksize),
        low_memory=False,
    ):
        chunk_index += 1
        remove = local_mask(chunk, args).to_numpy()
        if remove.any():
            counts = chunk.loc[remove, "FractureScale"].astype(str).value_counts()
            for key, value in counts.items():
                removed_counts[key] = removed_counts.get(key, 0) + int(value)
            removed_rows += int(remove.sum())
        kept = chunk.loc[~remove].copy()
        kept["Local200Replacement"] = 0
        kept.to_csv(
            temporary_csv,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig" if first_write else "utf-8",
            lineterminator="\n",
        )
        first_write = False
        kept_rows += len(kept)
        if chunk_index % 4 == 0:
            print(
                f"[local-replacement] chunks={chunk_index} "
                f"kept={kept_rows} removed={removed_rows}",
                flush=True,
            )

    local = local.reindex(columns=header.columns)
    local["Local200Replacement"] = 1
    if "PatchID" in local.columns:
        local["PatchID"] = (
            "local200__" + local["PatchID"].astype(str)
        )
    if "GenerationStage" in local.columns:
        local["GenerationStage"] = (
            local["GenerationStage"].astype(str)
            + "|local_200m_coverage"
        )
    local.to_csv(
        temporary_csv,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    temporary_csv.replace(output_csv)
    local_counts = {
        str(key): int(value)
        for key, value in local["FractureScale"].astype(str).value_counts().items()
    }
    summary = {
        "status": "pass",
        "large_patches_csv": str(large_csv),
        "local_patches_csv": str(local_csv),
        "output_csv": str(output_csv),
        "replacement_box": {
            "time_ms": [float(args.time_min_ms), float(args.time_max_ms)],
            "x_m": [float(args.x_min), float(args.x_max)],
            "y_m": [float(args.y_min), float(args.y_max)],
        },
        "removed_large_patch_counts": removed_counts,
        "removed_large_patch_count": int(removed_rows),
        "appended_local_patch_count": int(len(local)),
        "appended_local_patch_counts": local_counts,
        "kept_large_patch_count": int(kept_rows),
        "output_total_rows": int(kept_rows + len(local)),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[local-replacement] status=pass output={output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
