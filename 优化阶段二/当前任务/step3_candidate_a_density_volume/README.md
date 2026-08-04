# Step3 Candidate A Density Volume

当前目录把候选 A 区块的正式样本包空间化为局部裂缝密度体。

## 当前定位

主线固定为：

`裂缝密度样本 -> 裂缝密度体 -> 初始DFN -> 井控校正后的DFN`

这里实现的是第二个对象：`候选 A` 区块、`T4-T7` 层段内的局部裂缝密度体。

## 当前口径

- 输入样本：
  - [candidate_a_t4_t7_density_training_samples.csv](/home/tyh/projects/petroleum/code/GAN-DFN/优化阶段二/当前任务/step2_target_block_and_density_samples/output/candidate_a_t4_t7_density_training_samples.csv)
- 平面网格底座：
  - [trace_header_xy.csv](/home/tyh/projects/petroleum/code/GAN-DFN/数据精简/trace_header_xy.csv)
- 区块边界：
  - `X=[568567.32, 571567.32]`
  - `Y=[4199778.50, 4202778.50]`
- 输出层位：
  - `沙三段`
  - `沙四段`
- 空间化方法：
  - 对每个道点，在当前层段内对邻近样本做 `k` 近邻加权
  - 权重由 `1/distance * PointConfidence * source_weight` 组成
  - `real_well` 样本权重乘 `2.0`
  - `virtual_well` 样本权重乘 `1.0`

## 输出

- 局部裂缝密度体 CSV：
  - `candidate_a_t4_t7_density_volume.csv`
- 统计摘要：
  - `candidate_a_t4_t7_density_volume_summary.json`

## 用法

```bash
python 优化阶段二/当前任务/step3_candidate_a_density_volume/build_candidate_a_density_volume.py \
  --config 优化阶段二/当前任务/step3_candidate_a_density_volume/configs/candidate_a_density_volume.json
```

## 说明

- 当前输出先以 `CSV` 形式交付，便于核查和直接进入后续初始 DFN 规则生成。
- 当前不是全矿区体，而是候选 A 的局部正式体。
- 当前没有先走 `step4/step45` 关系校准，而是直接把正式 `DensityLabel` 空间化，优先形成可落地密度体对象。
