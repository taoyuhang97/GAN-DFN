# QC 规则说明

本批 QC 仅覆盖 unified point table 的最小验收门槛，定位是“先拦住明显坏表”，不替代后续建模前的专题核查。

## Error 级

- 缺少任一必需字段：
  - `WellName`
  - `X`
  - `Y`
  - `TIME`
  - `TVD`
  - `FractureDensity`
  - `IsFracturePoint`
  - `SourceType`
- `X/Y/TIME/TVD` 任一字段存在空值

## Warning 级

- `FractureDensity` 或关键坐标字段中存在不可转数值的记录
- `IsFracturePoint = 1` 但 `FractureDensity` 缺失
- `IsFracturePoint = 1` 但 `FractureDensity <= 0`
- `IsFracturePoint = 0` 但 `FractureDensity > 0`
- `FractureDensity < 0`
- `WellName/X/Y/TIME/TVD` 组合键出现重复点
- `SourceType` 分布不可统计

## 状态解释

- `pass`
  - 无 error、无 warning
- `warn`
  - 无 error，但存在 warning
- `fail`
  - 存在任一 error

## 当前默认假设

- `IsFracturePoint` 支持 `0/1/true/false/yes/no/y/n`
- `FractureDensity` 默认按非负连续值处理
- `SourceType` 允许多来源混合，但必须可统计占比

## 建议后续补充规则

- 按 `LayerGroup` 检查每层样本量下限
- 按 `SourceType` 检查真实井/虚拟井比例上限
- 检查 `PointConfidence/WellConfidence` 是否落在 `[0, 1]`
- 检查 `MD/TVD/TIME` 的井内单调性
