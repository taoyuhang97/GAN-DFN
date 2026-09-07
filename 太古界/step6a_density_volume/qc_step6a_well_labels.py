#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
import segyio
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
VOL = ROOT / "output/taigu_step6a_full_20260902/volume"
INPUT = (ROOT.parent / "step5_virtual_wells/output/taigu_step5_full_20260902/step5b/taigu_step5b_unified_samples.csv").resolve()

def main():
    with segyio.open(str(VOL / "predicted_fracture_density.sgy"), strict=False) as f:
        xs = np.asarray(f.attributes(segyio.TraceField.SourceX)[:], float)
        ys = np.asarray(f.attributes(segyio.TraceField.SourceY)[:], float)
        axis = np.asarray(f.samples, float)
        cube = np.stack([np.asarray(f.trace[i], dtype=np.float32) for i in range(f.tracecount)], axis=0)
    tree = cKDTree(np.column_stack([xs, ys]))
    all_df = pd.read_csv(INPUT, encoding="utf-8-sig")
    xy = all_df[["X", "Y"]].to_numpy(float)
    dist, idx = tree.query(xy, k=1)
    ti = np.rint((all_df["TIME"].to_numpy(float) - axis[0]) / 2.0).astype(int)
    valid = (ti >= 0) & (ti < len(axis)) & (dist <= 20)
    df = all_df.loc[valid].copy()
    df["PredictedDensity"] = cube[idx[valid], ti[valid]]
    df["TraceDistanceM"] = dist[valid]
    df["DensityTarget"] = pd.to_numeric(df["DensityLabel"], errors="coerce")
    df["PresenceLabel"] = pd.to_numeric(df["PresenceLabel"], errors="coerce")
    rows = []
    for (well, kind, layer), g in df.groupby(["SourceWellName", "SourceKind", "LayerGroup"]):
        p = g["PredictedDensity"].to_numpy(float); y = g["PresenceLabel"].to_numpy(float); d = g["DensityTarget"].to_numpy(float); pos = y == 1
        rows.append({"SourceWellName": well, "SourceKind": kind, "LayerGroup": layer, "Rows": len(g), "PositiveRows": int(pos.sum()), "PredMean": float(np.nanmean(p)), "PredP90": float(np.nanpercentile(p, 90)), "TargetDensityMean": float(np.nanmean(d[pos])) if pos.any() else np.nan, "PredAtPositiveMean": float(np.nanmean(p[pos])) if pos.any() else np.nan, "MAE_Positive": float(np.nanmean(np.abs(p[pos] - d[pos]))) if pos.any() else np.nan, "PresenceCorrelation": float(np.corrcoef(y, p)[0, 1]) if len(g) > 1 and np.std(y) > 0 and np.std(p) > 0 else np.nan, "MeanTraceDistanceM": float(np.mean(g["TraceDistanceM"]))})
    pd.DataFrame(rows).to_csv(VOL / "well_label_qc.csv", index=False, encoding="utf-8-sig")
    summary = {"input_rows": int(len(all_df)), "mapped_rows": int(len(df)), "mapping_rate": float(len(df) / len(all_df)), "groups": int(len(rows)), "pred_mean": float(np.nanmean(df["PredictedDensity"])), "pred_at_positive_mean": float(np.nanmean(df.loc[df["PresenceLabel"] == 1, "PredictedDensity"]))}
    (VOL / "well_label_qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
