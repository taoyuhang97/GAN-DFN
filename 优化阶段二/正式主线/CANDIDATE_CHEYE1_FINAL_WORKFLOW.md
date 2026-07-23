# 车页1导眼 demo 最终流程整理

## 当前结论

当前已经完成严格统一后缀重跑。

建议统一后缀：

`candidate_cheye1_final_lowcoh_vertical_v1`

最终剖面输出：

`step9_section_visualize/output/candidate_cheye1_final_lowcoh_vertical_v1_small_projection/cheye1_dfn_coherence_sections/`

## 严格最终流程

### Stage 0：输入坐标 QC

作用：
确认 demo 区地震道、属性体、层位、断层解释和三维密度体坐标一致。

代码：

`step6b_demo_density_volume_3d/qc_multiscale_rebalance_inputs.py`

当前可复用输出：

`step6b_demo_density_volume_3d/output/candidate_cheye1_multiscale_rebalance_v1/input_qc/`

说明：
该步骤只做 QC，不参与后续数值构造。当前 QC 已通过，可以作为最终流程的输入一致性证明。

### Step6A：小尺度背景密度体

作用：
从原始三维裂缝密度体中提取小尺度背景裂缝密度。

代码：

`step6b_demo_density_volume_3d/build_step6a_small_background.py`

当前可复用输出：

`step6b_demo_density_volume_3d/output/candidate_cheye1_multiscale_rebalance_v1/step6a_small/`

输入：

- `candidate_cheye1_3d_predicted_density.sgy`
- `candidate_cheye1_3d_trace_mapping.npz`

输出：

- `small_background_prior.npz`
- `small_background_qc.json`

说明：
Step6A 不直接使用相干体/蚂蚁体；它表达测井和虚拟井推演出的背景小尺度裂缝。当前版本可继续复用。

### Step6B：中尺度裂缝带先验

作用：
识别中尺度裂缝发育带。新版逻辑为“蚂蚁体分支 + 垂向低相干独立分支”。

代码：

`step6b_demo_density_volume_3d/build_step6b_medium_corridor_prior.py`

最终应使用输出：

`step6b_demo_density_volume_3d/output/candidate_cheye1_step6b_medium_lowcoh_vertical_v1/step6b_medium/`

输入：

- 原始三维裂缝密度体 trace mapping
- 相干体
- 蚂蚁体
- 最大曲率体
- 最大正曲率体

输出：

- `medium_corridor_prior.sgy`
- `medium_corridor_mask.sgy`
- `medium_corridor_components.npz`
- `medium_corridor_component_summary.csv`
- `medium_corridor_qc.json`

当前 QC：

- 保留中尺度组件：214 个
- 保留体素：75,527 个
- 组件倾角中位数：70.29 度
- 低相干垂向分支贡献的保留体素：65,969 个

说明：
这是当前中尺度部分的最终推荐版本。

### Step6C：大尺度断层/断裂带先验

作用：
处理甲方提供的原始断层解释结果，并补充少量可解释的大尺度低相干断裂候选。

代码：

`step6b_demo_density_volume_3d/build_step6c_large_fault_prior.py`

当前推荐输出：

`step6b_demo_density_volume_3d/output/candidate_cheye1_step6c_large_faultlike_surface_v1/step6c_large/`

输入：

- `FaultStick-GeoEast.dat`
- 单元分割断层面 patch
- 相干体等地震属性体

输出：

- `large_fault_prior.sgy`
- `large_fault_mask.sgy`
- `large_fault_prior_components.npz`
- `large_fault_component_summary.csv`
- `large_fault_qc.json`

说明：
当前大尺度以原始断层解释为硬约束。不要把水平连续低相干层位直接解释为大尺度断层。

### Step6D：多尺度证据整合

作用：
不是简单把小中大密度相加，而是构造 Step7A 所需的小尺度最终密度体：

- 背景小尺度裂缝来自 Step6A。
- 中尺度裂缝带周围生成中尺度损伤小裂缝密度。
- 大尺度断层周围生成大尺度损伤小裂缝密度。
- 中尺度和大尺度核心仍保留给 Step7B/Step7C，不在 Step7A 中重复生成。

代码：

`step6b_demo_density_volume_3d/build_step6d_multiscale_bundle.py`

严格最终版已重新生成：

`step6b_demo_density_volume_3d/output/candidate_cheye1_final_lowcoh_vertical_v1/step6d_bundle/`

输入应固定为：

- Step6A：`candidate_cheye1_multiscale_rebalance_v1/step6a_small`
- Step6B：`candidate_cheye1_step6b_medium_lowcoh_vertical_v1/step6b_medium`
- Step6C：`candidate_cheye1_step6c_large_faultlike_surface_v1/step6c_large`

输出：

- `background_small_density.sgy`
- `medium_damage_small_density.sgy`
- `large_damage_small_density.sgy`
- `final_small_density.sgy`
- `final_small_candidate_mask.sgy`
- `multiscale_prior_bundle.npz`
- `multiscale_bundle_summary.json`

说明：
这是当前最需要重新跑的步骤。旧的 Step7A 使用的是 `candidate_cheye1_multiscale_current_flow_v2/step6d_bundle`，没有和新版低相干垂向 Step6B 完全对齐。

当前结果：

- `status=pass`
- Step6A 输入：`candidate_cheye1_multiscale_rebalance_v1/step6a_small`
- Step6B 输入：`candidate_cheye1_step6b_medium_lowcoh_vertical_v1/step6b_medium`
- Step6C 输入：`candidate_cheye1_step6c_large_faultlike_surface_v1/step6c_large`
- `final_small_candidate_mask` 体素数：596,507
- medium damage 小尺度体素数：557,577
- large damage 小尺度体素数：462,588

### Step7A：小尺度 DFN

作用：
从 Step6D 输出的小尺度最终密度体生成背景小裂缝和中大尺度周围的衍生小裂缝。

代码：

`step7a_small_scale_dfn/build_small_scale_dfn.py`

严格最终版已重新生成：

`step7a_small_scale_dfn/output/candidate_cheye1_final_lowcoh_vertical_v1/`

输入应来自新的 Step6D：

- `background_small_density.sgy`
- `medium_damage_small_density.sgy`
- `large_damage_small_density.sgy`

输出：

- `small_dfn_patches.csv`
- `small_dfn_raw_time.vtk`
- `small_dfn_summary.json`

当前结果：

- `status=pass`
- 小尺度裂缝片：7,575 个
- 背景小裂缝：4,087 个
- 中尺度损伤衍生小裂缝：2,121 个
- 大尺度断层损伤衍生小裂缝：1,367 个

### Step7B：中尺度 DFN

作用：
从 Step6B 中尺度裂缝带组件生成中尺度裂缝片。

代码：

`step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py`

最终应使用配置：

`step7b_multiscale_initial_dfn/configs/formal_candidate_cheye1_step7b_medium_lowcoh_vertical_v1.json`

当前可复用输出：

`step7b_multiscale_initial_dfn/output/candidate_cheye1_step7b_medium_lowcoh_vertical_v1/`

输出：

- `medium_dfn_patches.csv`
- `medium_dfn_raw_time.vtk`
- `medium_dfn_summary.json`

当前 QC：

- 中尺度裂缝片：593 个
- 覆盖层内候选组件：95/99
- 倾角小于 30 度比例：0
- 倾角中位数：75.58 度

说明：
当前版本可作为最终中尺度版本复用。

### Step7C：大尺度 DFN

作用：
基于原始断层解释和大尺度先验生成大尺度断层面/影响带裂缝片。

代码：

`step7c_large_fault_dfn/build_large_fault_dfn.py`

当前推荐配置：

`step7c_large_fault_dfn/configs/formal_candidate_cheye1_step7c_large_from_step6c_faultlike_surface_v1.json`

当前可复用输出：

`step7c_large_fault_dfn/output/candidate_cheye1_step7c_large_faultlike_surface_v1/`

输出：

- `large_fault_and_damage_patches.csv`
- `large_fault_and_damage_raw_time.vtk`
- `large_fault_dfn_summary.json`

当前 QC：

- 大尺度裂缝片：228 个
- 状态：pass

说明：
当前版本可作为最终大尺度版本复用。

### Step7D：多尺度 DFN 融合

作用：
融合 Step7A/Step7B/Step7C 的小中大尺度裂缝片。

代码：

`step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py`

严格最终版已重新生成：

`step7d_multiscale_fused_dfn/output/candidate_cheye1_final_lowcoh_vertical_v1/`

输入应固定为：

- Step7A：`candidate_cheye1_final_lowcoh_vertical_v1/small_dfn_patches.csv`
- Step7B：`candidate_cheye1_step7b_medium_lowcoh_vertical_v1/medium_dfn_patches.csv`
- Step7C：`candidate_cheye1_step7c_large_faultlike_surface_v1/large_fault_and_damage_patches.csv`

输出：

- `fused_multiscale_dfn_patches.csv`
- `fused_multiscale_dfn_raw_time.vtk`
- `fused_multiscale_summary.json`

当前结果：

- `status=pass`
- 融合裂缝片：8,396 个
- small：7,575 个
- medium：593 个
- large：228 个

### Step8：测井纠偏

作用：
在融合 DFN 基础上加入测井约束。成像测井真实解释段优先使用真实裂缝标签，常规测井段使用预测裂缝点。

代码：

`step8_dfn_well_correction/correct_dfn_with_well_controls.py`

严格最终版已重新生成：

`step8_dfn_well_correction/output/candidate_cheye1_final_lowcoh_vertical_v1/`

输入：

- Step7D 最终融合 CSV
- Step4 常规测井预测裂缝点
- Step3 成像测井真实裂缝解释结果
- 层位文件

输出：

- `well_corrected_dfn_fracture_patches.csv`
- `well_corrected_dfn_raw_time.vtk`
- `well_corrected_dfn_summary.json`
- 调试 VTK

当前结果：

- `status=pass`
- 井控后裂缝片：9,124 个
- small：8,229 个
- medium：667 个
- large：228 个
- medium 倾角小于 30 度：0 个

### Step8 小尺度井控修正版

测井裂缝是井点尺度约束，不应直接生成或改写中大尺度结构。因此新增 `smallwell_v1` 修正版。

配置：

`step8_dfn_well_correction/configs/formal_well_control_correction_candidate_cheye1_final_lowcoh_vertical_v1_smallwell_v1.json`

输出：

`step8_dfn_well_correction/output/candidate_cheye1_final_lowcoh_vertical_v1_smallwell_v1/`

修改逻辑：

- 井控匹配只允许选择 `FractureScale=small` 的初始裂缝片。
- 无匹配时，新增井控裂缝片也只使用 small 模板。
- 所有 `IsWellControlPatch=1` 的裂缝统一标记为 `FractureScale=small`。
- 常规测井预测裂缝没有真实倾向倾角，不再直接继承模板陡倾角，而是采用“小尺度模板方向 + 分层目标倾角 + 稳定扰动”的方向策略。
- Step3 成像测井真实解释裂缝仍使用真实 `Frac_Azimuth / Frac_Dip`。

当前结果：

- `status=pass`
- 井控后裂缝片：9,128 个
- 井控裂缝片：768 个，全部为 small
- 常规测井预测井控片：708 个，倾角中位数约 63.64 度，倾角大于等于 75 度为 0 个
- 成像测井真实井控片：60 个，保留真实解释倾角

### Step9：剖面图

作用：
生成过车页1导眼的 DFN、相干体、二者叠加、局部 200m 叠加剖面。

代码：

`step9_section_visualize/build_cheye1_dfn_coherence_sections.py`

严格最终版已重新生成：

`step9_section_visualize/output/candidate_cheye1_final_lowcoh_vertical_v1_small_projection/cheye1_dfn_coherence_sections/`

小尺度井控修正版已重新生成：

`step9_section_visualize/output/candidate_cheye1_final_lowcoh_vertical_v1_smallwell_v1_small_projection/cheye1_dfn_coherence_sections/`

输入：

- Step8 最终 DFN VTK/CSV
- 相干体
- Step3 成像测井真实裂缝解释结果
- 原始断层解释 `FaultStick-GeoEast.dat`
- 层位文件

输出固定为 8 张 PNG：

- `01_dfn_section_xz_t4_t7.png`
- `02_dfn_section_yz_t4_t7.png`
- `03_coherence_section_xz_t4_t7.png`
- `04_coherence_section_yz_t4_t7.png`
- `05_coherence_dfn_overlay_xz_t4_t7.png`
- `06_coherence_dfn_overlay_yz_t4_t7.png`
- `07_coherence_dfn_overlay_200m_xz_t4_t7.png`
- `08_coherence_dfn_overlay_200m_yz_t4_t7.png`

说明：
Step9 当前使用小尺度投影显示逻辑：小尺度裂缝允许半宽投影，避免剖面过稀；中尺度和大尺度仍按裂缝片与剖面的真实交线显示。

## 当前最终结果

最终已完成统一后缀重跑：

1. Step6D：`candidate_cheye1_final_lowcoh_vertical_v1/step6d_bundle`
2. Step7A：`candidate_cheye1_final_lowcoh_vertical_v1`
3. Step7D：`candidate_cheye1_final_lowcoh_vertical_v1`
4. Step8：`candidate_cheye1_final_lowcoh_vertical_v1`
5. Step9：`candidate_cheye1_final_lowcoh_vertical_v1_small_projection`

Step6A、Step6B、Step6C、Step7B、Step7C 为复用的已确认结果，Step6D 之后均已按最终统一链路重跑。

## 2026-07-23 版本检查点补充

### Step8 事件聚合小尺度井控

本次固化默认采用：

`step8_dfn_well_correction/configs/formal_well_control_correction_candidate_cheye1_final_lowcoh_vertical_v1_smallwell_event_v1.json`

相对 `smallwell_v1`，该版本把 Step4 的 729 个密集预测点聚合为 104 个井尺度事件，最终使用 99 个 Step4 事件和 60 个 Step3 成像测井真值控制。输出为 8,537 个裂缝片，其中 159 个井控片全部为 small；相对 Step7D 初始 DFN 的 changed/added 比例约为 1.89%。

该版本作为本次代码检查点默认 Step8，但 `gap=6 ms`、`max_span=18 ms` 仍需在后续逐井、分层核对后决定是否调整。

### Step9 扩展

- small 裂缝允许限定半宽投影，中大尺度保持真实曲剖面交线。
- 井控 small 裂缝默认不参加半宽投影，避免重复强化井旁人工约束。
- 新增 AntTrack、CurvatureMax 和地震振幅过井曲剖面。
- 新增全矿区逐道 `TraceIdx/T4/T5/T6/T7` 层位表和曲剖面求交模块。
- 当前逐道层位只完成全矿区地震振幅预览；正式 DFN/相干体和属性背景图仍需下一阶段重跑验收。
- 原始断层解释已载入，但当前车页1导眼曲剖面与原始断层无足够交点，所以可见原始断层段数为 0。

### 保留的验证缺口

- Step6 尚未运行真正的留一井重新训练验证，不能据当前 QC 宣称充分跨井泛化。
- Step7B 已覆盖 95/99 个层内 component，但陡倾低相干体素精确中心命中率为 7.76%；需用完成逐道层位后的 Step9 图件判断是否继续调整 component 内采样。
