# Step 3 Imaging Supervision Samples

本步骤的正式成果是 `formal_rebuild`，用于构造成像测井强监督训练样本。

## 正式口径

- `Density` 是主监督标签。
- 外部成像井 `车660-1/车660-2/车662/车663` 的 `Density` 只取成像统计文件中的 `FVDC`。
- 矿区内成像井 `车151HF/车页1导眼` 的属性底板只取 Step2 正式样本，旧 labeled 样本只用于取 `P10` 标签来源。
- `HasFracture = Density > 0`，只是辅助存在标签。
- `GT_POINT_FLAG` 是原始裂缝点位按 `MD/DEPT` 映射到样本网格后的点位旗标，不等同于 `HasFracture`。
- 外部井补样必须与 Step2 主表和 `3x3_context` 同构；旧 `SEIS_0~SEIS_62` 只允许作为恢复 `SeisAmp` 九点上下文的中间来源，不作为正式输出字段。
- 缺失的 `Coherence/AntTrack/CurvatureMax/CurvaturePos` 及其 `3x3` 上下文置空，保留后续补体数据空间。
- 不生成冗余总训练表，训练输入只使用 `groups/*.csv`。

## 正式输入

- 矿区内底板：`step2_real_well_t4_t7_samples/output/formal_all_wells/{车151HF,车页1导眼}`
- 外部井 around/LAS/裂缝密度/裂缝点位：详见 `output/formal_rebuild/source_manifest.csv`

## 正式输出

- `output/formal_rebuild/groups/*.csv`
- `output/formal_rebuild/sample_group_manifest.csv`
- `output/formal_rebuild/source_manifest.csv`
- `output/formal_rebuild/formal_data_contract_columns.csv`
- `output/formal_rebuild/delivery_group_qc.csv`
- `output/formal_rebuild/delivery_outer_step2_contract_qc.csv`
- `output/formal_rebuild/delivery_acceptance_summary.json`

## 运行

```bash
python 优化阶段二/正式主线/step3_imaging_supervision_samples/build_formal_rebuild_groups.py
python 优化阶段二/正式主线/step3_imaging_supervision_samples/validate_formal_rebuild_delivery.py
```

验收要求：`delivery_acceptance_summary.json` 中 `status=pass`。

## 禁用成果

- `output/formal_imaging_supervision/imaging_supervision_main.csv` 不是当前正式 Step3 输入。
- `build_imaging_supervision_samples.py` 是旧总表式实现，不再作为当前正式主线入口。
- `step3_real_well_prediction` 是错误放入正式主线的调试基线，不能作为正式 Step3 或 Step4 输入。
