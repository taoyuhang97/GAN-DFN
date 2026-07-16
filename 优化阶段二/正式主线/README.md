# 优化阶段二 正式主线

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

## 当前最终 Step6

目录：`step6b_demo_density_volume_3d`

保留代码：

- `build_candidate_3d_density_sgy.py`：构造基础三维裂缝密度体，SGY 输出继承地震 trace header。
- `build_step6a_small_background.py`：Step6A，小尺度背景密度。
- `build_step6b_medium_corridor_prior.py`：Step6B，中尺度裂缝带先验，蚂蚁体为主，相干体/曲率体辅助。
- `build_step6c_large_fault_prior.py`：Step6C，大尺度断层/断裂带先验，原始断层 patch 为硬约束，低相干陡向异常仅作补充候选。
- `build_step6d_multiscale_bundle.py`：Step6D，整合 A/B/C，输出最终小尺度背景与损伤带衍生密度。
- `build_multiscale_density_bundle.py`：Step6A/B/C/D 共用 SGY、网格、属性体工具。
- `qc_step6_positioning.py`、`qc_multiscale_density_vs_attributes.py`、`qc_multiscale_rebalance_inputs.py`：输入坐标、属性体一致性和多尺度证据 QC。

保留配置：

- `configs/formal_candidate_cheye1_3d_density_sgy.json`：基础三维密度体构造。
- `configs/formal_candidate_cheye1_multiscale_density_v1.json`：Step6A/B/C/D 多尺度密度构造基础配置。

当前实际使用的 Step6D 输出：

```text
step6b_demo_density_volume_3d/output/candidate_cheye1_multiscale_current_flow_v2/step6d_bundle/
```

其中 Step7A 当前使用：

- `background_small_density.sgy`
- `medium_damage_small_density.sgy`
- `large_damage_small_density.sgy`

## 当前最终 Step7

### Step7A 小尺度裂缝

目录：`step7a_small_scale_dfn`

保留代码：

- `build_small_scale_dfn.py`

当前最终配置：

- `configs/formal_candidate_cheye1_step7a_small_current_flow_v5_v2_density_v4_orientation.json`

当前口径：

- 空间采样保留 v2 的高密度候选阈值和按 `SamplingWeight` 优先的抽样思想。
- 不再使用 `min_patch_count`、`max_patch_count`、`count_scale` 这类固定数量约束。
- 当前采用 `v2_weighted_probability`：每个候选体元按权重产生裂缝片，最终数量随密度体和区域规模自然变化。
- 倾向倾角使用 v4 地质方向族，不直接复制中大尺度裂缝/断层方向。

当前实际输出：

```text
step7a_small_scale_dfn/output/candidate_cheye1_current_flow_v5_v2_density_v4_orientation/
```

### Step7B 中尺度裂缝带

目录：`step7b_multiscale_initial_dfn`

保留代码：

- `build_medium_scale_dfn_v4.py`
- `build_multiscale_initial_dfn_preview.py`：历史预览工具，当前不作为主入口。

当前最终配置：

- `configs/formal_candidate_cheye1_step7b_medium_from_step6b_rebuild_v3_strict_steep30_preview.json`

当前口径：

- 中尺度裂缝来自 Step6B 识别的裂缝带/组件。
- 倾角下限采用 30°，避免大量近水平裂缝主导结果。
- 中尺度裂缝表达连续带主体，小尺度衍生裂缝由 Step7A 表达。

### Step7C 大尺度断层/断裂带

目录：`step7c_large_fault_dfn`

保留代码：

- `build_large_fault_dfn.py`

当前最终配置：

- `configs/formal_candidate_cheye1_step7c_large_from_step6c_faultlike_surface_v1.json`

当前口径：

- 原始单元断层面是硬约束，倾向倾角应尽量保持原始断层 patch/panel 的局部几何。
- 自识别低相干大尺度候选仅作为补充断裂带候选，不替代原始断层解释。

### Step7D 大中小尺度融合

目录：`step7d_multiscale_fused_dfn`

保留代码：

- `build_multiscale_fused_dfn.py`

当前最终配置：

- `configs/formal_candidate_cheye1_step7d_fused_current_flow_v1.json`

当前输入：

- Step7A：`candidate_cheye1_current_flow_v5_v2_density_v4_orientation`
- Step7B：`candidate_cheye1_step7b_medium_from_step6b_rebuild_v3_strict_steep30_preview`
- Step7C：`candidate_cheye1_step7c_large_faultlike_surface_v1`

## Step7B 初始三维工具说明

目录：`step7b_initial_dfn_3d`

该目录当前不再作为正式独立 Step7B 主线，但 `build_initial_dfn_from_3d_density_sgy.py` 仍被 Step7A 复用，用于裂缝片几何、VTK 导出、局部 PCA 方向估计等工具函数。因此保留代码文件，删除旧试参配置。

## Step8/Step9 当前配置

Step8 当前最终配置：

- `step8_dfn_well_correction/configs/formal_well_control_correction_candidate_cheye1_current_flow_v1.json`

Step9 当前最终配置：

- `step9_section_visualize/configs/formal_candidate_cheye1_current_flow_v1_with_original_fault_trace_dfn_coherence_sections.json`

Step9 仍保留通用矿区/井剖面属性体绘图配置，用于后续生成其它剖面图。

## 已清理的旧路径

以下旧路径已从代码主线删除：

- `step6_demo_density_volume`：早期二维密度路径。
- `step7_initial_dfn`：早期二维 DFN 细化路径。
- `step7c_fused_dfn`：早期二维/三维/断层后融合试验路径。
- 早期 `lowcoh`、`seismic_prior`、`rebalance`、`preview` 试参配置。

如需回看这些旧版本，可从提交 `eb36745` 或更早历史恢复。
