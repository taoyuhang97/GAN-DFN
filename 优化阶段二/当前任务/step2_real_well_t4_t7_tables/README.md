# 真实井 T4-T7 三表构造

这一步只保留真实井在 `T4-T7` 目标层段内的井段点，不保留 `T4` 以上和 `T7` 以下井段。

输出固定为三张表：

- `*_t4_t7_real_well_main.csv`
  - 点级主表，保留 `T4-T7` 目标井段的连续主段点位
  - 包含：`SampleID/WellName/X/Y/TIME/TVD/DEPT`
  - 包含：标准测井曲线 `AC/CAL/CNL/DEN/GR/RFOC/RILD/RILM/SP`
  - 包含：5 类体属性中心值 `SeisAmp/Coherence/AntTrack/CurvatureMax/CurvaturePos`
  - 包含：5 类体属性井周统计特征 `Mean/Std/Min/Max/ValidCount`
  - 包含：样本可用性字段 `SampleUsableForModel/SampleUsableStatus`

- `*_t4_t7_real_well_3x3_context.csv`
  - `3x3` 邻域宽表
  - 每个样本点一行，保留 `X/Y/TIME`
  - 各体属性按 `x0_y0 ~ x2_y2` 展开为 `3x3` 九点属性列

- `*_t4_t7_real_well_interval.csv`
  - 井级层位边界表
  - 每井一行
  - 只保留：`T4/T5/T6/T7` 分界点的 `DEPT/TVD/TIME`

- `*_t4_t7_real_well_qc_summary.csv`
  - 井级质控汇总表
  - 每井一行
  - 只保留：目标井段点数、裁头裁尾点数、是否存在中间非法点、最终主表点数、最终可建模点数

当前口径：

- 层位挂接复用 `step0_t4_t7_surface_tools`
- 井旁属性按真实 `X/Y/TIME` 插值
- 平面邻域按 `12.5m` 网格定义 `3x3`
- 支持直井和斜井
- 井名别名处理：`导眼井 -> 车页1导眼`，`车页1HF -> 车页1导眼`
- 异常体值不再删点，而是保留连续井段、将坏值置空并在主表中打可用性状态
- `T4-T7` 井段先按时间落在 `T4-T7` 内筛出目标层段，再保留最长连续主段
- 只允许对连续主段头尾非法点做裁剪，不在主段内部挖洞

示例运行：

```bash
python 优化阶段二/当前任务/step2_real_well_t4_t7_tables/build_real_well_t4_t7_tables.py \
  --config 优化阶段二/当前任务/step2_real_well_t4_t7_tables/configs/demo_57_576.json \
  --max-points-per-well 300 \
  --max-log-rows-per-well 300 \
  --max-workers 2
```

全井批量：

```bash
python 优化阶段二/当前任务/step2_real_well_t4_t7_tables/build_real_well_t4_t7_tables.py \
  --config 优化阶段二/当前任务/step2_real_well_t4_t7_tables/configs/all_las_wells.json \
  --max-workers 4
```
