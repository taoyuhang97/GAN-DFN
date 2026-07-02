# Step 4 Expert Real-Well Prediction

本步骤基于 Step 3 formal_rebuild group 样本训练地层先分库、井级专家实体的正式 Step 4，并部署到 Step 2 全井 `T4-T7` 正式样本。

当前输入契约：所有现有测井属性与井旁属性统一作为模型输入，缺失列按 `NaN` 处理，不再为单个专家裁剪特征子集。

硬口径：

- `Density` 是主监督。
- `HasFracture` 和点位只用于 Stage 1 辅助概率和后处理细化。
- 验证按井留一，`车页1导眼` 只做 holdout，不进入训练专家。
- 输入使用 Step 3 `formal_rebuild/groups/*.csv` 和 Step 2 `formal_all_wells` 真实井底板。
- `step3_real_well_prediction` 仍视为调试基线，不作为正式成果来源。

运行：

```bash
python 优化阶段二/正式主线/step4_expert_real_well_prediction/train_and_predict_expert_library.py \
  --config 优化阶段二/正式主线/step4_expert_real_well_prediction/configs/formal_expert_prediction.json
```

主要输出：

- `output/formal_well_expert_library/input_group_manifest.csv`
- `output/formal_well_expert_library/feature_contract.csv`
- `output/formal_well_expert_library/train_holdout_split_manifest.csv`
- `output/formal_well_expert_library/expert_library/expert_registry.csv`
- `output/formal_well_expert_library/expert_library/expert_signature.csv`
- `output/formal_well_expert_library/expert_library/loo_validation_metrics.csv`
- `output/formal_well_expert_library/well_strata_expert_match_manifest.csv`
- `output/formal_well_expert_library/all_wells_t4_t7_density_prediction.csv`
- `output/formal_well_expert_library/all_wells_t4_t7_fracture_points.csv`
- `output/formal_well_expert_library/all_wells_t4_t7_prediction_summary.csv`
- `output/formal_well_expert_library/validation/*`
- `output/formal_well_expert_library/real_well_predictions/<well>/*`

下游契约：

- `all_wells_t4_t7_density_prediction.csv` 必须保留 `Density` 和 `HasFracture` 标准列，供 Step 5 单源井虚拟测井直接读取。
- `Density` 是 `PredDensity` 的正式下游别名，表示真实井连续裂缝密度预测结果。
- `HasFracture` 是 `PredHasFracture` 的正式下游别名，只作为辅助存在标记。
- `step4_acceptance_summary.json` 中 `downstream_contract.checks` 必须全部为 `true`，否则不得进入 Step 5。
