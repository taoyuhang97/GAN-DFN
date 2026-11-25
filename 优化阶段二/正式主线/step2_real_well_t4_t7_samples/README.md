# Step 2 Real-Well T4-T7 Samples

## 常规测井多曲线主表（2026-11-23）

新增入口 `build_multilog_curve_samples.py`。每口井生成
`<Well>_t4_t7_real_well_conventional_curves.csv`，主表只保留 11 列：

```text
DEPT, GR, AC, CNL, DEN, LLD, LLS, RD, RS, RILD, RILM
```

LAS 路径、测次 PassID、实际曲线别名、覆盖区间、读取错误和构建说明不写入主表，
分别保存在 `formal_all_wells/` 下的
`conventional_curve_source_manifest.csv`、
`conventional_curve_source_intervals.csv`、
`multilog_curve_build_summary.csv`、
`multilog_curve_read_errors.csv` 和 `multilog_curve_build_summary.json`。

模型阶段只使用六类原始曲线；稳健 Z、分位秩、梯度和局部残差等派生特征在 Step4 内计算，
不把数据来源字段作为模型特征。

当前 LAS 读取兼容 V1.2 的 `~A` 数据段，并将 `HAC/DT24` 识别为 `AC`、
`CN/NPHI/CNCF` 识别为 `CNL`、`RHOB` 识别为 `DEN`。同一 LAS 中重复的深度列
只保留第一有效列。

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

GR/电阻率补充输出：

- `*_t4_t7_real_well_gr_resistivity.csv`
  - 由独立脚本读取现有主表的 `DEPT` 网格生成，不修改上述四张正式表
  - 固定字段：`DEPT/GR_LLD_LLS/LLD/LLS/GR_RD_RS/RD/RS/GR_RILD_RILM/RILD/RILM`
  - 同一曲线对的多个 LAS 可分段补充；重叠区按覆盖与采样质量择一，不取平均
  - 单个 LAS 内和拼接边界只允许跨越不超过 `0.5m` 的小空缺，大空缺保留为空值
  - 不按数值大小删除电阻率，不使用饱和值阈值；仅将非数值、NaN 和 LAS 明确声明的 `NULL` 视为缺失
  - 数值截断、对数变换和模型侧异常值处理在 Step4 完成

- `gr_resistivity_build_summary.csv`
  - 根目录井级汇总，记录每组曲线使用的文件、有效行、缺失行、重叠行和冲突行

- `gr_resistivity_read_errors.csv`
  - 原始 LAS 读取错误；无错误时保留固定表头的空表

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

GR/电阻率全井补充命令：

```bash
bash 优化阶段二/正式主线/step2_real_well_t4_t7_samples/run_formal_gr_resistivity_enrichment.sh
```
