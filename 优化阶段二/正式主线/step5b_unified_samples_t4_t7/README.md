# Step 5B Unified T4-T7 Density Samples

本步骤把 Step 4 真实井连续 `Density` 预测结果与 Step 5A 单源虚拟井弱监督样本合并，形成供 Step 6 demo 密度体训练使用的统一 `T4-T7` 样本表。

## 正式输入

- Step 4 真实井预测：`step4_expert_real_well_prediction/output/formal_well_expert_library/all_wells_t4_t7_density_prediction.csv`
- Step 4 真实井裂缝点：`step4_expert_real_well_prediction/output/formal_well_expert_library/all_wells_t4_t7_fracture_points.csv`
- Step 5A 虚拟井训练样本：`step5a_single_source_virtual_wells/output/formal_single_source_virtual_wells/virtual_well_training_samples.csv`
- Step 5A 虚拟井汇总：`step5a_single_source_virtual_wells/output/formal_single_source_virtual_wells/virtual_well_training_summary.csv`

## 正式输出

- `output/unified_t4_t7_density_samples.csv`
- `output/unified_t4_t7_density_samples_summary.json`
- `output/unified_t4_t7_field_contract.csv`
- `output/unified_t4_t7_qc.csv`

## 运行

```bash
python 优化阶段二/正式主线/step5b_unified_samples_t4_t7/build_unified_t4_t7_density_samples.py \
  --config 优化阶段二/正式主线/step5b_unified_samples_t4_t7/configs/formal_unified_t4_t7_density_samples.json
```

## 下游关系

`step6_demo_density_volume` 只读取本步骤输出的 `unified_t4_t7_density_samples.csv`，不直接读取 Step 5A 的虚拟井中间表。
