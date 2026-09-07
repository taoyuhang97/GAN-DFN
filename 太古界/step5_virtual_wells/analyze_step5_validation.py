#!/usr/bin/env python3
"""Summarize full-well Step5 validation by well, layer and depth bin."""
import argparse, json
from pathlib import Path
import pandas as pd

ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--step4-csv',type=Path,required=True); a=ap.parse_args()
o=a.output_dir; idx=pd.read_csv(o/'taigu_step5_virtual_well_index.csv',encoding='utf-8-sig'); cov=pd.read_csv(o/'taigu_step5_coverage_qc.csv',encoding='utf-8-sig'); s=pd.read_csv(a.step4_csv,encoding='utf-8-sig'); s=s[s.PredictionValid.astype(int)==1].copy()
s['TVD']=pd.to_numeric(s['TVD'],errors='coerce'); s['SourceRowIndex']=s.groupby('WellName').cumcount()
src=s[['WellName','SourceRowIndex','StrataName','TVD']].rename(columns={'WellName':'SourceWellName','TVD':'SourceTVD'})
idx=idx.merge(src,on=['SourceWellName','SourceRowIndex'],how='left'); idx['DepthBin']=pd.cut(idx.SourceTVD, bins=5, labels=['Q1_shallow','Q2','Q3','Q4','Q5_deep'])
idx['Accepted']=idx.TrainingEligible.astype(int)
def tab(keys):
 g=idx.groupby(keys,dropna=False).agg(candidate_count=('Accepted','size'),accepted_count=('Accepted','sum'),accepted_rate=('Accepted','mean')).reset_index(); return g
well=tab(['SourceWellName']); layer=tab(['SourceWellName','StrataName']); depth=tab(['SourceWellName','StrataName','DepthBin'])
well.to_csv(o/'step5_validation_by_well.csv',index=False,encoding='utf-8-sig'); layer.to_csv(o/'step5_validation_by_layer.csv',index=False,encoding='utf-8-sig'); depth.to_csv(o/'step5_validation_by_depth.csv',index=False,encoding='utf-8-sig')
reason=idx.groupby(['SourceWellName','StrataName','ExcludeReason']).size().reset_index(name='count'); reason.to_csv(o/'step5_validation_reasons_by_well_layer.csv',index=False,encoding='utf-8-sig')
summary={'wells':int(idx.SourceWellName.nunique()),'candidate_count':int(len(idx)),'accepted_count':int(idx.Accepted.sum()),'accepted_rate':float(idx.Accepted.mean()),'coverage_source_points':cov.groupby('InOBNCoverage').size().to_dict(),'reason_counts':idx.ExcludeReason.value_counts().to_dict(),'well_rate_min':float(well.accepted_rate.min()),'well_rate_max':float(well.accepted_rate.max()),'well_rate_mean':float(well.accepted_rate.mean()),'layer_rate_min':float(layer.accepted_rate.min()),'layer_rate_max':float(layer.accepted_rate.max())}
(o/'step5_validation_stability_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))
