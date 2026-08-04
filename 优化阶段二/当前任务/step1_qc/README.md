# Step1 QC

本目录提供 unified point table 的最小质量检查脚本与验收说明，目标是先把第三步统一点表的入口质量门槛固定下来。

## 文件

- `qc_unified_point_table.py`
  - 主脚本
  - 输入 `csv/parquet`
  - 输出 `qc_summary.json`、`qc_summary.csv`、`source_distribution.csv`

## 最小输入字段

当前默认必需字段：

- `WellName`
- `X`
- `Y`
- `TIME`
- `TVD`
- `FractureDensity`
- `IsFracturePoint`
- `SourceType`

## 检查项

脚本当前会做以下最小 QC：

1. 必需字段是否存在
2. `X/Y/TIME/TVD` 是否为空
3. `FractureDensity` 与 `IsFracturePoint` 的基本一致性
4. `SourceType` 来源分布
5. 扩展检查：
   - 数值列是否可转为数值
   - `WellName/X/Y/TIME/TVD` 组合键是否存在重复点

### 一致性规则

- `IsFracturePoint = 1` 的记录不应缺失 `FractureDensity`
- `IsFracturePoint = 1` 的记录建议 `FractureDensity > 0`
- `IsFracturePoint = 0` 的记录若 `FractureDensity > 0`，会记为警告
- `FractureDensity < 0` 记为警告

## 运行方式

```bash
python 优化阶段二/当前任务/step1_qc/qc_unified_point_table.py \
  --input /path/to/unified_point_table.csv \
  --output-dir /path/to/qc_output
```

严格模式：

```bash
python 优化阶段二/当前任务/step1_qc/qc_unified_point_table.py \
  --input /path/to/unified_point_table.csv \
  --output-dir /path/to/qc_output \
  --strict
```

## 输出说明

- `qc_summary.json`
  - 面向程序消费
  - 包含总状态、错误数、警告数、来源分布、运行参数
- `qc_summary.csv`
  - 面向人工验收
  - 同时包含总览指标和逐项检查结果
- `source_distribution.csv`
  - `SourceType` 的条数与占比

## 建议扩展字段

如果 unified point table 后续逐步稳定，建议补充以下字段以增强 QC 和下游可解释性：

- `SampleID`
- `LayerGroup`
- `MD`
- `PointConfidence`
- `WellConfidence`
- `SourceWellName`
- `VirtualWellName`
- `FractureDensityWeak`
- `DensitySourceType`
- `UpdateTime`
