# 优化阶段二 正式主线

## 当前10 km修改入口（2026-07-28）

`formal_demo_10km_multiscale_flow_v1`已经完成工程跑通，但诊断发现时间轴、逐道T4-T7层窗、中尺度证据权重、大尺度成面和Step9全区过滤问题，当前结果不得作为最终地质DFN或对外图件。

下一轮唯一问题台账和执行入口：

- `DFN_10KM_ISSUES_AND_REFACTOR_PLAN_20260728.md`
- `CHECKPOINT_20260728_10KM_MULTISCALE_FLOW.md`

> 2026-07-30强制状态：当前仍有四项未关闭需求，分别为地震属性权重与裂缝空白、成像测井展示与井周校正、完整且不按层位截断的原始断层面、曲率体与变密度体灰度配色。详细状态和验收标准见问题台账第3.1节。任何脚本或全链路`pass`均不得替代这四项人工业务验收。

当前计划从层位合同修正开始，使用新版本`formal_demo_10km_multiscale_flow_v2`重跑，不覆盖v1问题节点。

本目录保留第二轮优化当前确认的正式代码。当前主线已经从早期“二维密度 + 三维密度再融合”的试验路径，整理为多尺度地震解释约束流程：

```text
Step1-5 样本准备
  -> Step6 基础三维密度体与多尺度地震解释先验
  -> Step7A 小尺度裂缝片
  -> Step7B 中尺度裂缝带裂缝片
  -> Step7C 大尺度断层/断裂带裂缝片
  -> Step7D 大中小尺度 DFN 融合
  -> Step8 井控纠偏
  -> Step9 剖面展示与地质解释一致性检查
```

输出目录 `**/output/` 不纳入 Git；代码、配置和文档纳入版本管理。

## 当前解释口径

- 蚂蚁体高值表示中尺度裂缝/断裂发育更强，低值表示不发育。
- 相干体低值表示地震不连续增强，可辅助识别断层/断裂带；但横向连续低相干条带常对应层界面，不应直接解释为有效裂缝。
- 曲率体主要作为小尺度裂缝背景和局部构造扰动辅助依据。
- 原始断层解释成果是大尺度断层硬约束，不能被低相干自识别候选替代。

## 2026-07-23 版本检查点

本次固化的 DFN 主体后缀为 `candidate_cheye1_final_lowcoh_vertical_v1`。Step8 默认检查点采用事件聚合小尺度井控 `smallwell_event_v1`，Step9 对应使用 `smallwell_event_v1_small_projection`。完整谱系见：

- `CANDIDATE_CHEYE1_FINAL_WORKFLOW.md`
- `CANDIDATE_CHEYE1_VERSION_MANIFEST.json`
- `../../docs/正式主线项目现状与后续修改_2026-07-23.md`

该检查点用于后续对比和继续开发，不表示所有科学问题均已关闭。当前仍需补做 Step6 真正的留一井重训练验证，并完成 Step9 逐道层位图件的全链路重跑和视觉验收。

## 当前 Step6

目录：`step6b_demo_density_volume_3d`

保留代码：

- `build_candidate_3d_density_sgy.py`：构造基础三维裂缝密度体，SGY 输出继承地震 trace header。
- `build_step6a_small_background.py`：Step6A，小尺度背景密度。
- `build_step6b_medium_corridor_prior.py`：Step6B，中尺度裂缝带先验，使用蚂蚁体主分支和独立陡倾低相干分支，曲率作为辅助支持。
- `build_step6c_large_fault_prior.py`：Step6C，大尺度断层/断裂带先验，原始断层 patch 为硬约束，低相干陡向异常仅作补充候选。
- `build_step6d_multiscale_bundle.py`：Step6D，整合 A/B/C，输出最终小尺度背景与损伤带衍生密度。
- `build_multiscale_density_bundle.py`：Step6A/B/C/D 共用 SGY、网格、属性体工具。
- `qc_step6_positioning.py`、`qc_multiscale_density_vs_attributes.py`、`qc_multiscale_rebalance_inputs.py`：输入坐标、属性体一致性和多尺度证据 QC。

基础配置：

- `configs/formal_candidate_cheye1_3d_density_sgy.json`：基础三维密度体构造。
- `configs/formal_candidate_cheye1_multiscale_density_v1.json`：Step6A/B/C/D 多尺度密度构造基础配置。

当前检查点使用的 Step6 输入谱系：

```text
Step6A: output/candidate_cheye1_multiscale_rebalance_v1/step6a_small/
Step6B: output/candidate_cheye1_step6b_medium_lowcoh_vertical_v1/step6b_medium/
Step6C: output/candidate_cheye1_step6c_large_faultlike_surface_v1/step6c_large/
Step6D: output/candidate_cheye1_final_lowcoh_vertical_v1/step6d_bundle/
```

Step6D 输出给 Step7A 的三个小尺度密度域为：

- `background_small_density.sgy`
- `medium_damage_small_density.sgy`
- `large_damage_small_density.sgy`

## 当前 Step7

### Step7A 小尺度裂缝

目录：`step7a_small_scale_dfn`

保留代码：

- `build_small_scale_dfn.py`

当前检查点配置：

- `configs/formal_candidate_cheye1_step7a_small_final_lowcoh_vertical_v1.json`

当前口径：

- 空间采样保留 v2 的高密度候选阈值和按 `SamplingWeight` 优先的抽样思想。
- 不再使用 `min_patch_count`、`max_patch_count`、`count_scale` 这类固定数量约束。
- 当前采用 `v2_weighted_probability`：每个候选体元按权重产生裂缝片，最终数量随密度体和区域规模自然变化。
- 倾向倾角使用 v4 地质方向族，不直接复制中大尺度裂缝/断层方向。

当前输出：

```text
step7a_small_scale_dfn/output/candidate_cheye1_final_lowcoh_vertical_v1/
```

### Step7B 中尺度裂缝带

目录：`step7b_multiscale_initial_dfn`

保留代码：

- `build_medium_scale_dfn_v4.py`
- `build_multiscale_initial_dfn_preview.py`：历史预览工具，当前不作为主入口。

当前检查点配置：

- `configs/formal_candidate_cheye1_step7b_medium_lowcoh_vertical_v1.json`

当前口径：

- 中尺度裂缝来自 Step6B 识别的裂缝带/组件。
- 倾角下限采用 30°，避免大量近水平裂缝主导结果。
- 中尺度裂缝表达连续带主体，小尺度衍生裂缝由 Step7A 表达。
- 当前覆盖 95/99 个层内 component，但陡倾低相干体素的精确中心命中率为 7.76%。component 覆盖率与体素精确贴合不是同一指标，后续需结合 Step9 图件判断是否继续调整采样位置。

### Step7C 大尺度断层/断裂带

目录：`step7c_large_fault_dfn`

保留代码：

- `build_large_fault_dfn.py`

当前检查点配置：

- `configs/formal_candidate_cheye1_step7c_large_from_step6c_faultlike_surface_v1.json`

当前口径：

- `large_original_fault_merged_surface_raw_time.vtk` 保存 Step6C 选中单元断层直接拼接后的曲面，仅用于原始地质解释对比。
- `large_fault_surface_raw_time.vtk` 与 `large_fault_surface_patches.csv` 保存逐道 T4-T7 内的原始断层 surface fragments，作为最终 DFN 中的原始断层硬约束。
- `large_inferred_fault_surfaces_raw_time.vtk` 保存 Step6C 相干体主导、自主识别的补充小断层，不替代原始断层解释。
- `large_fault_damage_zone_raw_time.vtk` 仅为损伤带诊断结果，不进入 Step7D；损伤带衍生小裂缝由 Step6D/Step7A 表达。
- Step7D 正式读取 `large_fault_dfn_patches.csv`，其中只包含原始断层 fragments 和自主识别断层。

### Step7D 大中小尺度融合

目录：`step7d_multiscale_fused_dfn`

保留代码：

- `build_multiscale_fused_dfn.py`

当前检查点配置：

- `configs/formal_candidate_cheye1_step7d_fused_final_lowcoh_vertical_v1.json`

当前输入：

- Step7A：`candidate_cheye1_final_lowcoh_vertical_v1`
- Step7B：`candidate_cheye1_step7b_medium_lowcoh_vertical_v1`
- Step7C：`candidate_cheye1_step7c_large_faultlike_surface_v1`

## Step7B 初始三维工具说明

目录：`step7b_initial_dfn_3d`

该目录当前不再作为正式独立 Step7B 主线，但 `build_initial_dfn_from_3d_density_sgy.py` 仍被 Step7A 复用，用于裂缝片几何、VTK 导出、局部 PCA 方向估计等工具函数。因此保留代码文件，删除旧试参配置。

## Step8 当前检查点

配置：

- `step8_dfn_well_correction/configs/formal_well_control_correction_candidate_cheye1_final_lowcoh_vertical_v1_smallwell_event_v1.json`

当前逻辑：

- Step3 成像测井真值保留真实方向。
- Step4 常规测井密集点按 `gap=6 ms`、`max_span=18 ms` 聚合为事件。
- 所有井控裂缝限制为 small，不改变中大尺度构造主体。
- 当前输出 8,537 个裂缝片，其中 159 个井控片全部为 small；相对 Step7D 的 changed/added 比例约 1.89%。

## Step9 当前检查点

DFN/相干体 8 图配置：

- `step9_section_visualize/configs/formal_candidate_cheye1_final_lowcoh_vertical_v1_smallwell_event_v1_small_projection_dfn_coherence_sections.json`

当前口径：

- 中、大尺度裂缝只显示真实曲剖面交线。
- small 裂缝允许限定半宽投影；井控 small 裂缝默认不参与该投影。
- 原始断层解释已载入，但当前车页1导眼 XZ/YZ 曲剖面与原始断层没有足够交点，因此可见原始断层段数为 0；这不表示断层输入缺失。
- 地质属性、地震振幅和逐道 T4-T7 层位工具已纳入代码。逐道层位目前只完成全矿区振幅预览，其他正式图件仍需在下一阶段重跑验收。

Step9 仍保留通用矿区/井剖面属性体绘图配置，用于后续生成其它剖面图。

## 已清理的旧路径

以下旧路径已从代码主线删除：

- `step6_demo_density_volume`：早期二维密度路径。
- `step7_initial_dfn`：早期二维 DFN 细化路径。
- `step7c_fused_dfn`：早期二维/三维/断层后融合试验路径。
- 早期 `lowcoh`、`seismic_prior`、`rebalance`、`preview` 试参配置。

如需回看这些旧版本，可从提交 `eb36745` 或更早历史恢复。
