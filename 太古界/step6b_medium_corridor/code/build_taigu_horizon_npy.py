import csv
from pathlib import Path
import numpy as np
src=Path(__file__).resolve().parents[2]/'common/horizon_contract/output/taigu_attribute_full_horizon_contract_v1/demo_horizon_contract.csv'
dst=src.with_suffix('.npy'); rows=[]
with src.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):
    def v(k,d=np.nan):
      try:return float(r[k]) if r[k] else d
      except:return d
    ok=int(v('SurfaceValid',0)); rows.append((int(r['TraceIdx']),v('TopTimeMs'),v('MidTimeMs'),v('MidTimeMs'),v('BaseTimeMs'),ok,ok,ok,0))
dt=np.dtype([('TraceIdx','<i4'),('T4','<f4'),('T5','<f4'),('T6','<f4'),('T7','<f4'),('SurfaceOrderValid','u1'),('ShasanPresent','u1'),('ShasiPresent','u1'),('HorizonCorrectionCode','u1')])
np.save(dst,np.asarray(rows,dtype=dt)); print(dst)
