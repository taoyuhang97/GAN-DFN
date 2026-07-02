# Step 2 Real-Well T4-T7 Samples

本步骤生成矿区内真实井在 `T4-T7` 目标层段内的正式四表样本，作为后续所有实验的底板输入。

正式输出四张表：

- `*_t4_t7_real_well_main.csv`
  - 点级主表
  - 保留：`SampleID/WellName/X/Y/TIME/TVD/DEPT`
  - 保留：常规测井 `AC/CAL/CNL/DEN/GR/RFOC/RILD/RILM/SP`
  - 保留：5 类井周中心属性 `SeisAmp/Coherence/AntTrack/CurvatureMax/CurvaturePos`
  - 保留：5 类井周统计特征 `Mean/Std/Min/Max/ValidCount`
  - 保留：样本可用性字段 `SampleUsableForModel/SampleUsableStatus`

- `*_t4_t7_real_well_3x3_context.csv`
  - 点级 `3x3` 井旁属性宽表
  - 保留：`SampleID/WellName/X/Y/TIME`
  - 保留：5 类井周体属性按 `x0_y0 ~ x2_y2` 展开的九宫格原值

- `*_t4_t7_real_well_interval.csv`
  - 井级层位边界表
  - 每井一行
  - 保留：`T4/T5/T6/T7` 分界点的 `DEPT/TVD/TIME`

- `*_t4_t7_real_well_qc_summary.csv`
  - 井级质控汇总表
  - 每井一行
  - 保留：目标井段点数、裁头裁尾点数、是否有内部非法点、最终主表点数、最终可建模点数

当前正式口径：

- 只处理 `T4-T7`
- 先按真实井点 `X/Y/TIME` 挂接层位
- 井周属性按真实点位插值，不贴最近道
- 平面邻域按 `12.5m` 做 `3x3`
- 目标井段按 `T4-T7` 时间窗筛选后保留最长连续主段
- 只允许对主段头尾非法点裁剪，不在主段中间删点

全井批量命令：

```bash
python 优化阶段二/正式主线/step2_real_well_t4_t7_samples/build_real_well_t4_t7_samples.py \
  --config 优化阶段二/正式主线/step2_real_well_t4_t7_samples/configs/formal_all_wells.json \
  --max-workers 4
```
