# 训练样本打包

本目录用于将单元 DFN 与对应地震窗口打包为研究内容三可直接使用的训练样本。

## 脚本

- `build_instance_gan_dataset.py`
  - 旧版稠密标签打包脚本，仅用于保留历史实验逻辑，不建议继续使用。
- `build_sparse_instance_gan_dataset.py`
  - 轻量版正式脚本。
  - 输出固定窗口地震输入体与稀疏裂缝实例标签。
  - 目标是替代旧版 `target_geom.npy` 的稠密保存方式。

## 轻量版输出内容

每个窗口保存一个 `.npz` 样本包，核心字段包括：

- `input_features`
- `valid_z_mask`
- `instance_ijk`
- `instance_geom`
- `instance_source`
- `instance_weight`
- `instance_patch_index`
- `count_volume`（可选）

聚合目录下保存：

- `sample_manifest.csv`
- `unit_manifest.csv`
- `dataset_summary.json`
- `skipped_units.csv`

## 默认输出目录

- `/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/GAN训练准备/训练样本打包`

## 推荐起步参数

```powershell
py -3.12 .\研究内容三\优化阶段一\GAN训练准备\训练样本打包\build_sparse_instance_gan_dataset.py `
  --stats-run-dir /data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/GAN训练准备/窗口级实例统计/full_20260330_t128_t160_ov50 `
  --run-name phase1_t128_sparse_v1 `
  --window-size 128 `
  --overlap-ratio 0.5 `
  --input-channels seismic_amp grad_x grad_y grad_z rel_depth_in_interval
```
