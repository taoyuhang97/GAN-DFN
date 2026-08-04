# Step2 T4-T7 Density Package

这个目录提供“正式多井 T4-T7 Density 训练样本包构建入口”，输入直接对接阶段二正式交付的 `01` 真实井连续裂缝密度曲线，输出可直接进入密度体建模的样本表和统计摘要。

## 目标

- 扫描 `01` 目录下多井 `分井连续裂缝密度曲线`
- 复用 `step0_t4_t7_surface_tools` 的真实层位挂接逻辑
- 按 `XY` 最近曼哈顿距离挂接真实 `T4/T5/T6/T7` 层位面
- 仅保留 `T4-T7` 目标层段且层位顺序/厚度检查通过的样本
- 输出正式汇总样本 CSV 和 summary JSON

当前固定层位分组口径：

- `T4 <= TIME < T6` => `沙三段`
- `T6 <= TIME < T7` => `沙四段`

## 文件

- `build_t4_t7_density_sample_package.py`
  - 正式多井批量构建脚本
- `configs/real_wells_t4_t7_official.json`
  - 对接阶段二正式 `01` 目录的示例配置
- `output/`
  - 默认输出目录

## 用法

在仓库根目录执行：

```bash
python 优化阶段二/当前任务/step2_t4_t7_density_package/build_t4_t7_density_sample_package.py \
  --config 优化阶段二/当前任务/step2_t4_t7_density_package/configs/real_wells_t4_t7_official.json
```

## 输出

默认生成：

- `优化阶段二/当前任务/step2_t4_t7_density_package/output/real_wells_t4_t7_density_samples.csv`
- `优化阶段二/当前任务/step2_t4_t7_density_package/output/real_wells_t4_t7_density_samples_summary.json`

样本表核心字段包括：

- `SourceFile`
- `WellName`
- `X`
- `Y`
- `TIME`
- `TVD`
- `DEPT`
- `Density`
- `DensityRaw`
- `HasFracture`
- `LayerGroup`
- `LayerGroupCode`
- `T4Time`
- `T5Time`
- `T6Time`
- `T7Time`
- `T4_MANHATTAN_DISTANCE`
- `T5_MANHATTAN_DISTANCE`
- `T6_MANHATTAN_DISTANCE`
- `T7_MANHATTAN_DISTANCE`

summary JSON 至少包含：

- `used_well_count`
- `record_count`
- `density_stats`
- `records_per_well`
- `layer_group_distribution`
- `selected_surface_files`

## Density 口径

- 当前项目 `Density` 是主监督字段
- `HasFracture` 只作为辅助字段和缺失补零规则依据
- 对 `HasFracture=0` 且 `Density` 为空的行，脚本会标准化为 `Density=0.0`
- 同时保留原始字段副本 `DensityRaw`
- 若出现 `HasFracture=1` 但 `Density` 仍为空，则该行不会进入最终样本，并在 summary 中统计

## 已知限制

- 当前不做 `3000m` 区块自动筛选
- 当前只构建真实井 `01` 样本包，不合并 `02` 虚拟井外推样本
- 当前不做 DFN 校正控制包联动
