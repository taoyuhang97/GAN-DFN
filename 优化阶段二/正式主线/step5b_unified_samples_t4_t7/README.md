# Step 5B 统一 T4-T7 双标签样本

本步骤把当前 Step4 v3 真实井合并预测与 Step5A 普通曲率主导的虚拟井样本合并，生成后续 Step6 使用的唯一正式训练表。

## 输入

- Step4 v3 `six_expert_prediction_manifest.json` 指定的最终合并预测和裂缝点表。
- Step2 已对齐的真实井主表，用 `WellName + DEPT` 给 Step4 真实井行回接地震属性。
- Step5A `formal_curvature_led_v2/virtual_well_training_samples.csv`。

禁止直接填写旧 Step4 聚合表路径，也禁止使用旧 Step5A 结果。

## 标签和属性合同

- `PresenceLabel` 是二值裂缝存在标签。
- `DensityLabel` 是仅对正样本定义的条件连续密度；负样本必须保持空值，不能 `fillna(0)`。
- 真实井 `PointConfidence=1`、`SampleWeight=1`；虚拟井保留Step5A置信度和按25条轨迹归一化后的权重。
- 正式地震属性为 `SeisAmp`、`Coherence`、`AntTrack`、`CurvatureMax`；普通曲率必须有效。
- `CurvaturePos` 不属于正式合同。
- 为支持当前Step6读取和后续按源井分组验证，统一表只保留16列：`SourceKind`、`SourceWellName`、`TrackWellName`、`X/Y/TIME`、`LayerGroup`、双标签、`HasFracture`兼容列、`PointConfidence`、`SampleWeight`和4种地震属性。
- DEPT/TVD、SampleID、专家/曲线对、支持段、虚拟时间映射、连续性分数、3×3统计量和其他重复字段不进入统一训练表；需要追查时通过Step4原表、Step5A索引和JSON审计完成。

## 验收

脚本检查：Step4 合并键唯一、虚拟源样点可回指当前真实样点、所有训练行存在普通曲率、低可信虚拟行已排除、正样本条件密度完整、负样本条件密度为空，以及虚拟总权重不超过配置允许的真实井总权重比例。

## 输出

新结果写入 `output/formal_curvature_led_v2`：

- `unified_t4_t7_density_samples.csv`
- `unified_t4_t7_density_samples_summary.json`
- `unified_t4_t7_field_contract.csv`
- `unified_t4_t7_qc.csv`

旧 `output` 下的历史统一样本不覆盖，避免新旧合同混用。

## 运行

```bash
python 优化阶段二/正式主线/step5b_unified_samples_t4_t7/build_unified_t4_t7_density_samples.py \
  --config 优化阶段二/正式主线/step5b_unified_samples_t4_t7/configs/formal_unified_t4_t7_density_samples.json
```
