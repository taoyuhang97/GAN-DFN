# Step1 Unified Samples

当前目录提供“统一井轨迹点级样本表”最小可运行脚手架，当前优先面向 `车页1导眼` 的正式输入，但通过 JSON 配置可扩展到其他井。

## 文件

- `build_unified_point_table.py`
  - 读取 `around_data.csv`、`fractures.csv`、`final_log_with_fractures.csv`、`final_fracture_points.csv`、`final_fracture_segments.csv`、`final_strata_segmentation.csv`
  - 输出统一点级样本表 csv
- `configs/cheye1_official.json`
  - `车页1导眼` 正式输入配置
- `output/`
  - 生成后的统一表和简要 summary

## 用法

在仓库根目录执行：

```bash
python 优化阶段二/当前任务/step1_unified_samples/build_unified_point_table.py \
  --config 优化阶段二/当前任务/step1_unified_samples/configs/cheye1_official.json
```

输出文件：

- `优化阶段二/当前任务/step1_unified_samples/output/车页1导眼_unified_point_table.csv`
- `优化阶段二/当前任务/step1_unified_samples/output/车页1导眼_unified_point_table_summary.json`

## 当前实现口径

- 主表使用 `final_log_with_fractures.csv`
  - 保留其原始列，避免丢失上游语义
- `around_data.csv`
  - 按 `TVD` 最近邻挂接
  - 重点补回常见属性 `AC/DEN/GR/RHOB/SP`
- `final_strata_segmentation.csv`
  - 按深度区间挂接分层字段
- `final_fracture_segments.csv`
  - 按裂缝段深度区间挂接段级密度字段
- `final_fracture_points.csv`
  - 用 `WellName + TVD + TIME + X + Y` 复合键挂接点级预测字段
- `fractures.csv`
  - 按 `MD` 到主表 `TVD` 最近邻挂接，生成原始裂缝点标记和产状字段

## 统一表核心字段

- 来源与主键
  - `UnifiedRowID`
  - `ConfigName`
  - `BaseRowSource`
  - `Source*File`
- 基础坐标
  - `WellName`
  - `TVD`
  - `DEPT`
  - `TIME`
  - `X`
  - `Y`
- 常见属性
  - `AttrAC`
  - `AttrDEN`
  - `AttrGR`
  - `AttrRHOB`
  - `AttrSP`
- 裂缝点/密度
  - `CommonFractureFlag`
  - `CommonFractureDensity`
  - `RawFracturePointFlag`
  - `RawFracturePointCount`
  - `PredPointMatchFlag`
  - `PredSegmentMatchFlag`
- 分层
  - `CommonStrataName`
  - `StrataGeoSegmentID`
  - `StrataGeoIntervalKey`
  - `StrataGeoDepthMin`
  - `StrataGeoDepthMax`

## 公共接口字段

当前输出会在保留原始详细列的同时，额外补齐一组面向 `step1_qc` 和下游统一消费的公共接口列：

- `SourceWellName`
  - 优先取 `WellName`，缺失时回退 `StrataWellName`，再回退配置里的 `well_name`
- `SourceType`
  - 当前 `车页1导眼` MVP 固定写为 `well_trajectory_point_mvp`
  - 语义是“当前统一点表来自 MVP 版井轨迹点骨架”，便于 QC 统计来源占比
- `TVDRef`
  - 优先取 `TVD`，缺失时回退 `DEPT`
  - 当前 step1 统一按主表深度列作为公共参考深度
- `LayerGroup`
  - 优先取 `StrataAssignmentResolvedName`
  - 若为空，依次回退到 `CommonStrataName`、`StrataName`、`InterpStrataName`、`StrataProvidedName`
  - 仍为空时写 `UNSPECIFIED`
- `FractureDensity`
  - 优先对齐当前主标签密度列 `CommonFractureDensity`
  - 若该列缺失，再回退 `PredDensityMassPerLength`、`PredPointDensityMassPerLength`
  - 当前 MVP 中，对没有预测裂缝密度的点统一补 `0.0`
- `IsFracturePoint`
  - 明确按点级 `0/1` 输出
  - 当前口径对齐主标签：`CommonFractureFlag=1` 或 `FractureDensity>0` 时记为 `1`，否则记为 `0`
  - 原始成像裂缝点信息仍保留在 `RawFracturePoint*` 详细列
- `PointConfidence`
  - 仅对 `IsFracturePoint=1` 的点输出置信度
  - 优先取 `PredPointOrientationConfidence`，再回退 `PredSegmentOrientationConfidence`
  - 若正样本仍无现成置信度，补 `1.0`；负样本统一记 `0.0`

## 缺失字段规则

- 当前 6 个正式输入内没有直接提供“真实裂缝密度真值”
  - `MissingGTDensity` 留空
  - `MissingGTDensityNote` 标注原因
- 当前未接入断层/属性体/局部网格字段
  - `MissingFaultConstraint` 留空
  - `MissingFaultConstraintNote` 标注为后续步骤预留

## 已知限制

- `fractures.csv` 只有 `MD + 产状`，没有逐点裂缝密度，所以这里只能作为原始裂缝点监督挂接，不能直接生成真值密度。
- `around_data.csv` 与 `final_log_with_fractures.csv` 采样步长不同，当前按 `TVD` 最近邻补值，并保留 `AroundMatchDistance` 供下游检查。
- 当前默认 `fractures.csv` 的 `MD` 可直接按深度邻近映射到主表。若后续确认需要严格区分 `MD/TVD` 语义，应调整此处映射规则。
