# G-DFN监督基线

本目录用于在正式进入 GAN 之前，先完成“地震窗口 -> 裂缝实例/裂缝片”的监督重建基线。

## 目录说明

- `baseline_common.py`
  - 训练与推理共用函数
  - 包括样本清单读取、稀疏标签在线展开、窗口解码、单元回拼
- `baseline_model.py`
  - 3D U-Net 监督 baseline
- `build_unit_level_split.py`
  - 按 `UnitID` 做 train/val/test 划分
- `train_supervised_baseline.py`
  - 训练监督 baseline
- `infer_and_evaluate_baseline.py`
  - 用 checkpoint 做窗口预测、单元回拼、patch 对比评估

## 默认输入

- 轻量版训练样本：
  - `E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容三\GAN训练准备\训练样本打包\phase1_t128_sparse_v1_full_fix1_20260330`

## 默认输出

- `E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容三\G_DFN监督基线`

## 推荐执行顺序

1. 先做 Unit 级数据切分

```powershell
py -3.12 .\研究内容三\优化阶段一\G_DFN监督基线\build_unit_level_split.py `
  --dataset-run-dir E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容三\GAN训练准备\训练样本打包\phase1_t128_sparse_v1_full_fix1_20260330 `
  --run-name phase1_unit_split_v1
```

2. 再做监督 baseline 训练

```powershell
py -3.12 .\研究内容三\优化阶段一\G_DFN监督基线\train_supervised_baseline.py `
  --split-run-dir E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容三\G_DFN监督基线\数据切分\phase1_unit_split_v1 `
  --run-name baseline_v1
```

3. 最后做推理与单元级评估

```powershell
py -3.12 .\研究内容三\优化阶段一\G_DFN监督基线\infer_and_evaluate_baseline.py `
  --split-run-dir E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容三\G_DFN监督基线\数据切分\phase1_unit_split_v1 `
  --checkpoint E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容三\G_DFN监督基线\训练结果\baseline_v1\checkpoints\best_model.pt `
  --run-name baseline_eval_v1
```
