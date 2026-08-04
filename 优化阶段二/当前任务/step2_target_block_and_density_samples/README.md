# Step2 Target Block And Density Samples

当前目录承接两件正式事项：

1. 固定 `3000m x 3000m` 的 T4-T7 目标区块；
2. 基于 `01 + 02` 正式交付，构建可直接进入密度体建模的多井正式训练样本包。

## 当前默认口径

- 默认目标区块：候选 `A`
  - `X=[568567.32, 571567.32]`
  - `Y=[4199778.50, 4202778.50]`
- 默认正式井名单：
  - `车49`
  - `车57`
  - `车58`
  - `车71`
  - `车571`
  - `车571-3`
  - `车575`
  - `车斜576`
  - `车斜577`
- 当前层位筛选口径：
  - `T4 <= TIME < T6 => 沙三段`
  - `T6 <= TIME < T7 => 沙四段`
- 当前标签口径：
  - `02` 井周外推密度体是主监督样本主体
  - `01` 真实井连续密度曲线是同区块真实井硬约束
  - `Density` 是唯一主监督
  - 原始 `Density` 空值在样本包中统一落为 `DensityLabel=0.0`
  - `HasFracture` 只保留为辅助字段，不作为与 `Density` 同等级主标签

## 文件

- `build_density_training_samples.py`
  - 多井正式样本包构建脚本，统一并入 `01` 真实井和 `02` 虚拟井近井体
- `configs/candidate_a_realwells.json`
  - 候选 A 区块正式配置
- `output/`
  - 样本 CSV、summary JSON、区块选择摘要

## 用法

在仓库根目录执行：

```bash
python 优化阶段二/当前任务/step2_target_block_and_density_samples/build_density_training_samples.py \
  --config 优化阶段二/当前任务/step2_target_block_and_density_samples/configs/candidate_a_realwells.json
```

## 输出

- 正式样本表：
  - `优化阶段二/当前任务/step2_target_block_and_density_samples/output/candidate_a_t4_t7_density_training_samples.csv`
- 统计摘要：
  - `优化阶段二/当前任务/step2_target_block_and_density_samples/output/candidate_a_t4_t7_density_training_samples_summary.json`

## 训练字段建议

- 进入模型：
  - 当前统一后的字段名以 `02` 为主，`01` 中文属性列会映射为标准名
  - `X`
  - `Y`
  - `TIME`
  - `LayerGroup`
  - `SEIS_TRUE`
  - `COHERENCE`
  - `ANT_TRACK`
  - `CURVATURE_MAX`
  - `CURVATURE_POS`
- 辅助保留但不直接进模型：
  - `SourceKind`
  - `SourceWellName`
  - `TrackWellName`
  - `WellName`
  - `TVD`
  - `DEPT`
  - `HasFracture`
  - `PredAzimuth`
  - `PredDip`
  - `PredStrataName`
  - `PredSegmentID`
  - `T4Time`
  - `T5Time`
  - `T6Time`
  - `T7Time`
  - `InTargetT4T7`
  - `SurfaceOrderValid`
  - `InputRowValid`
  - 各 `*_MANHATTAN_DISTANCE`
