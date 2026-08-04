# 第四步到第五步桥接 Demo

本目录用于把第四步输出的“属性-裂缝密度关系表”，进一步转成第五步可直接使用的“局部裂缝发育控制场”。

当前版本只解决一件事：

```text
第三步统一样本表 + 第四步关系表 -> 样点控制分数 -> 局部控制场
```

它不是最终的区域裂缝发育带识别程序，也不负责生成最终 DFN。它的定位是：

- 固定第四步到第五步之间的对象接口；
- 验证“关系校准结果”是否能重新落到空间样本上；
- 为后续规则法、局部插值法、局部 GAN 提供统一入口。

## 1. 输入文件

需要三份输入。

### 1.1 第三步统一样本表

建议为 `csv`，一行表示一个 `X/Y/T` 样点。最少应包含：

- `LayerGroup`
- `X`
- `Y`
- `TIME`
- `FractureDensityWeak` 或 `FractureDensity`
- `PointConfidence`

推荐同时包含：

- `WellConfidence`
- `SEIS_TRUE`
- `COHERENCE`
- `ANT_TRACK`
- `CURVATURE_MAX`
- `CURVATURE_POS`
- `FRACTURE_INV`
- `FaultDistance`

### 1.2 第四步关系明细表

即 `relation_bins.csv`，来自：

- [优化阶段二/研究内容二/step4_demo/run_step4_demo.py](/home/tyh/projects/petroleum/code/GAN-DFN/优化阶段二/研究内容二/step4_demo/run_step4_demo.py:1)

### 1.3 第四步关系摘要表

即 `relation_summary.csv`，同样来自第四步 demo。

## 2. 当前桥接逻辑

第一步，读取第四步的分层属性关系。

- 按 `layer_group + attribute` 组织
- 只保留 `status == ok` 的关系
- 用 `|weighted_pearson|` 作为该属性在本层的关系权重

第二步，把第四步关系重新映射到第三步样本表。

- 对每个样点、每个属性，根据该属性当前值落入的分箱
- 取出对应分箱的 `weighted_fracture_density_mean`
- 作为这个属性对该样点的局部裂缝发育评分

第三步，对同一样点的多个属性评分做加权融合。

- 属性权重来自第四步摘要中的 `|weighted_pearson|`
- 样点最终控制分数是多属性评分的加权平均
- 再与第三步已有的 `FractureDensityWeak` 做 50/50 融合

第四步，把样点控制分数聚合为空间控制场。

- 按 `LayerGroup + X + Y + TIME` 分组
- 对同一位置的多个来源样本做可信度加权平均
- 输出局部 `ControlScore`

第五步，对控制分数做最小分类。

- `>= 0.65` 记为 `high`
- `[0.45, 0.65)` 记为 `medium`
- `< 0.45` 记为 `low`

这个分类只是 demo 阶段的局部控制标签，不是最终矿区尺度阈值方案。

## 3. 运行方式

```bash
python 优化阶段二/研究内容三/step45_bridge_demo/run_step45_bridge_demo.py \
  --sample-csv /path/to/unified_samples.csv \
  --relation-bins-csv /path/to/step4_output/relation_bins.csv \
  --relation-summary-csv /path/to/step4_output/relation_summary.csv \
  --output-dir /path/to/step45_output
```

如需提高关系筛选强度，可以调：

```bash
python 优化阶段二/研究内容三/step45_bridge_demo/run_step45_bridge_demo.py \
  --sample-csv /path/to/unified_samples.csv \
  --relation-bins-csv /path/to/step4_output/relation_bins.csv \
  --relation-summary-csv /path/to/step4_output/relation_summary.csv \
  --output-dir /path/to/step45_output \
  --min-attr-weight 0.10 \
  --min-valid-attributes 2
```

## 4. 输出文件

输出目录下至少包含：

- `scored_samples.csv`
  - 每个样点的多属性局部评分、融合分数、有效属性数
- `local_control_field.csv`
  - 局部 `X/Y/T` 控制场
- `control_field_summary.json`
  - 本次运行的统计摘要

## 5. 当前版本的边界

- 只做局部样本级控制场，不做全区三维体生成
- 不进行空间插值或形态学连通域分析
- 不做断层几何外推，只把 `FaultDistance` 当作可选普通属性
- 不替代第五步正式的“区域差异和裂缝发育带识别”

## 6. 为什么这样设计

这样做的好处是：

- 先把第四步结果真正落回空间样本上，而不是停留在表格层面
- 形成稳定的第五步输入对象：`local_control_field.csv`
- 后续若要升级成 GPU patch 模型，可以直接把局部控制场切成小块输入
- 后续若要接局部 GAN，也可以把 `ControlScore` 当作条件通道之一
