#!/usr/bin/env python3
"""Build the Step7A demo grid with TraceIdx from the attribute trace header."""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path('/home/tyh/projects/petroleum/code/GAN-DFN')
old_path = ROOT / '太古界/common/trace_contract/output/taigu_demo_grid_v1/demo_grid.csv'
attr_path = ROOT / '太古界/common/attribute_trace_contract/output/taigu_attribute_trace_header_v1/attribute_trace_header.csv'
out_path = ROOT / '太古界/step7a_small_scale_dfn/output/taigu_attribute_demo_grid_v3/demo_grid.csv'

old = pd.read_csv(old_path, encoding='utf-8-sig')
attr = pd.read_csv(attr_path, encoding='utf-8-sig')
tree = cKDTree(attr[['X', 'Y']].to_numpy(dtype=float))
dist, pos = tree.query(old[['X', 'Y']].to_numpy(dtype=float), k=1)
if float(dist.max()) > 2.0 or len(np.unique(pos)) != len(pos):
    raise RuntimeError(f'attribute grid nearest mapping invalid: max_distance={dist.max()}, unique={len(np.unique(pos))}/{len(pos)}')
out = old.copy()
out['TraceIdx'] = attr.iloc[pos]['TraceIdx'].to_numpy(dtype=np.int64)
out = out[['TraceIdx', 'X', 'Y', 'IX', 'IY']]
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'wrote {out_path} rows={len(out)} max_xy_distance_m={float(dist.max()):.3f}')
