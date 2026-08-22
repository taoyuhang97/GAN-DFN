# Step 2 TaiGuJie Regular-Log Samples

太古界实际入口是 `build_taigu_regular_samples.py`（v3，配置
`configs/taigu_step2_contracts_v3.json`），原有的
`build_real_well_t4_t7_samples.py` 和 `build_gr_resistivity_samples.py`
保留为砂砾岩正式主线参考，不能直接用于太古界。v1/v2 早期结果已清除，
当前正式输出为 `output/taigu_step2_regular_v3`。

## 输入与职责

- 深度语义：解释段上下界/密度/点位 = TVD；LAS DEPT = MD；
- 逐行轨迹：井斜文件 > LAS DEV/AZIM 积分 > 直井回退，逐行输出 `TVD/X/Y`；
- 常规井：读取 Step1 的本井 MD-TIME 层位合同，在 LAS 原始 MD 行上附加
  `TVD/X/Y/TIME/StrataName`，按**逐行 X/Y 挂层位**（每行查三界面时间）分类；
- 成像井：读取 Step1 的 TVD 合同，`StrataName` 按合同并裁剪到解释段
  TVD 区间；`TIME` 用临时借用常规井时深标注
  （`temporary_neighbor_time_depth`），只作标注、不作层位门控；
- 曲线列按 `GR > GR1 > GRSL` 优先选取，实际使用列写入段清单；
- 输出分离：段数据文件只有 `MD/TVD/X/Y/TIME/StrataName/GR/RD/RS`；井级/段级字段
  （WellName、SourcePath、LogDate、借用时深等）放入
  `taigu_step2_well_metadata.csv` 与 `taigu_step2_segment_manifest.csv`；
  无目标数据的井写入 `taigu_step2_rejected_wells.csv` 并说明原因；
- 段清单记录 `RawRowsRead/CompleteRatio`（完整 GR/RD/RS 行占比，不进主表）；
- 不读取地震体、不构造空间属性、不生成裂缝密度或点位标签。

常规井将本井时深文件视为连续分段线性函数：在本井时深最浅、最深 MD 之间插值，
不外推到该范围之外。LAS 的原始采样间隔不作为断段条件。

常规井只从 `侧向电阻率曲线/常规测井/RD` 读取；成像井只从
`侧向电阻率曲线/成像测井/RD` 读取。文件名必须与完整井名匹配，例如 `埕北30` 不会
匹配 `埕北301/303/305`。同井多 LAS **不合并**：每个 LAS 来源独立成段文件，
只保留其原始、完整的 `GR/RD/RS` 行，重叠测次原样保留，既不平均、不插值、
也不互斥剔除。每个段文件对应一段清单记录（来源、测年、MD 范围、行数、
实际曲线列、借用时深等），预测阶段再按规则合并。

## 埕北313特殊规则

埕北313走成像井 TVD 合同，地层界面为 4783 m（TVD）。同时它仅保留已确认的两个连续解释窗口：

```text
4174-4627 m
4751-4884 m
```

窗口之间的 `4627-4751 m`（TVD）作为真实中断段，输出中没有样点；不得在该段插值、填零或
拼接前后两段。该规则配置在 `configs/taigu_step2_contracts_v3.json`。

## 运行

```bash
python 太古界/step2_well_log_segments/build_taigu_regular_samples.py --replace-output
```

输出目录由配置指定，当前为 `output/taigu_step2_regular_v3`。

目录结构与正式主线的按井隔离方式一致：

```text
output/taigu_step2_regular_v3/
  埕北313/
    埕北313_seg_001.csv
    埕北313_seg_002.csv
    埕北313_seg_003.csv
  埕北39/
    埕北39_seg_001.csv
  .../
  taigu_step2_segment_manifest.csv
  taigu_step2_well_metadata.csv
  taigu_step2_rejected_wells.csv
  taigu_step2_acceptance_summary.json
```

根目录不保存所有井拼在一起的测井数据表。每口井的每个 LAS 来源（可再按覆盖中断
拆分）是一个段数据文件；地层归属保留在段数据文件的 `StrataName` 列。后续模型
必须以 `taigu_step2_segment_manifest.csv` 为入口逐段读取。

## 与正式主线的对照

| 项目 | 砂砾岩正式主线 | 太古界 Step2 | 判定 |
|---|---|---|---|
| 输出组织 | 每口井独立主表、区间表、QC | 每井每 LAS 来源独立段文件 + 根目录清单/元数据/放弃表 | v3 分离井级与段级字段 |
| 多 LAS | 独立 GR/电阻率补充脚本按覆盖优先级合并 | 各 LAS 独立成段、重叠保留，预测后再合并 | v3 不在 Step2 做不可逆选源 |
| 深度网格 | 主样点表的 `DEPT` 网格 | 原始 LAS 的 `MD` 网格 + 逐行 `TVD/X/Y` | v3 增加轨迹与 TVD |
| 层位挂接 | 逐样点 X/Y 挂 T4-T7 | 常规井逐行 X/Y 挂三界面；成像井按 TVD 合同 | v3 与砂砾岩对齐 |
| 时深 | 井点进入时间层位与地震属性 | 常规 LAS 本井时深；成像井借用时深仅作 TIME 标注 | v2；成像井层位仍走人工 MD 合同 |
| 地震与空间属性 | 读取 X/Y/TIME、地震体和 3x3 邻域 | 不读取 | 合理，当前模型只用 GR/RD/RS |

常规 RD 目录中没有完整 `GR/RD/RS` 覆盖的井不会输出，例如埕北30；这是按当前特征
合同主动排除，不是脚本读取失败。

# Historical Sandstone Reference

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
