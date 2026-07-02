# Step 5A Single-Source Virtual Wells

本步骤基于 Step 4 真实井连续 `Density` 预测结果，构造单源井虚拟测井弱监督样本。

硬口径：

- 虚拟测井是样本扩充层，不是实际井，也不是 DFN 单元。
- 每条虚拟井只允许有一个 `SourceWellName`。
- 虚拟点属性必须按自身 `X/Y/TIME` 从体数据重新取值。
- `Density` 由单源井对应点和属性连续性控制。
- 距离只控制候选范围和置信度，不进入 `Density` 主公式。
- 禁止继承旧 `copy_and_distance_decay` 弱标签逻辑。

运行：

```bash
python 优化阶段二/正式主线/step5a_single_source_virtual_wells/build_single_source_virtual_wells.py \
  --config 优化阶段二/正式主线/step5a_single_source_virtual_wells/configs/formal_single_source_virtual_wells.json
```

快速冒烟：

```bash
python 优化阶段二/正式主线/step5a_single_source_virtual_wells/build_single_source_virtual_wells.py \
  --config 优化阶段二/正式主线/step5a_single_source_virtual_wells/configs/formal_single_source_virtual_wells.json \
  --max-source-wells 1 --max-points-per-well 20
```

下游关系：

- 本步骤只负责生成单源虚拟井弱监督样本。
- 其正式下游是 `step5b_unified_samples_t4_t7`，由后者把真实井预测样本与本步骤虚拟井样本合并成统一训练表。

主要输出：

- `virtual_well_index.csv`
- `virtual_well_point_attributes.csv`
- `virtual_well_3x3_context.csv`
- `virtual_well_density_weaklabel.csv`
- `virtual_well_confidence.csv`
- `virtual_well_training_samples.csv`
- `virtual_well_training_summary.csv`
- `virtual_well_build_audit.json`
