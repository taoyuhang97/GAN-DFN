#!/usr/bin/env python3
"""Validate/export the Taigu Step5B unified training contract.

The legacy Taigu builder still computes real, imaging and virtual rows in one
pass.  This Step5B entry point makes the formal-mainline boundary explicit by
accepting its unified CSV, validating it, and writing a versioned Step5B copy.
It is intentionally conservative: it never silently repairs labels or fills
missing density with zero.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ["SourceKind", "SourceWellName", "TrackWellName", "X", "Y", "TIME", "LayerGroup", "PresenceLabel", "DensityLabel", "PointConfidence", "SampleWeight", "Coherence", "AntTrack", "CurvatureMax"]

def main() -> int:
    ap = argparse.ArgumentParser(description="Validate Taigu Step5B unified samples")
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    missing = [c for c in REQUIRED if c not in df.columns]
    checks = {"required_columns": not missing, "nonempty": len(df) > 0}
    if not missing:
        p = pd.to_numeric(df["PresenceLabel"], errors="coerce")
        d = pd.to_numeric(df["DensityLabel"], errors="coerce")
        kinds = set(df["SourceKind"].astype(str))
        anttrack = pd.to_numeric(df["AntTrack"], errors="coerce")
        checks.update({"binary_presence": bool(p.isin([0,1]).all()), "density_only_on_positive": bool(d[p.eq(0)].isna().all() and d[p.eq(1)].notna().all()), "positive_weights": bool(pd.to_numeric(df["SampleWeight"], errors="coerce").gt(0).all()), "finite_curvature": bool(pd.to_numeric(df["CurvatureMax"], errors="coerce").notna().all()), "finite_anttrack": bool(anttrack.notna().all()), "anttrack_minus_one_retained": bool(np.isclose(anttrack.to_numpy(dtype=float), -1.0, atol=1.0e-6).any()), "real_and_virtual_present": bool(bool(kinds.intersection({"weak_real","real_well"})) and "weak_virtual" in kinds)})
    out = args.output_dir / "taigu_step5b_unified_samples.csv"; summary_path = args.output_dir / "taigu_step5b_acceptance_summary.json"
    if out.exists() or summary_path.exists(): raise FileExistsError("Step5B output exists; choose a new versioned directory")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    summary = {"status": "pass" if all(bool(v) for v in checks.values()) and not missing else "fail", "input_csv": str(args.input_csv.resolve()), "output_csv": str(out.resolve()), "rows": int(len(df)), "missing_columns": missing, "checks": checks}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0 if summary["status"] == "pass" else 1

if __name__ == "__main__": raise SystemExit(main())
