# Step6B-Step7A/B/C/D 多尺度 DFN 改造实现文档

## 目标

本次改造目标不是继续把所有裂缝都压成一个统一密度体后随机生成裂缝片，而是把小、中、大三类裂缝分开处理，再统一融合：

- 小尺度裂缝：对应测井尺度和背景裂缝，由现有 Step6 三维密度体控制。
- 中尺度裂缝：对应蚂蚁体高值、局部低相干和曲率异常共同支持的带状裂缝发育区，由 Step6B 提取。
- 大尺度断层/断裂带：对应原始断层解释成果和陡倾低相干异常，由 Step6C 提取。
- 最终 DFN：Step7A/B/C 分尺度生成，Step7D 融合，随后进入 Step8 测井纠偏和 Step9 剖面展示。

本次所有新结果使用新目录和新后缀，不覆盖旧版 `lowcoh_steep`、`preview_v1/v2/v3`、旧 `step7/step7b/step7c` 输出。

## 总体输入

- Step6 原始三维裂缝密度体：
  `step6b_demo_density_volume_3d/output/candidate_cheye1/candidate_cheye1_3d_predicted_density.sgy`
- trace mapping：
  `step6b_demo_density_volume_3d/output/candidate_cheye1/candidate_cheye1_3d_trace_mapping.npz`
- 相干体：
  `/data/shared/project-oil/wx数据/砂砾岩/补充材料-20260623/车西_T4-T7/车西-相干体T4-T7.sgy`
- 蚂蚁体：
  `/data/shared/project-oil/wx数据/砂砾岩/补充材料-20260623/车西_T4-T7/车西-蚂蚁体T4-T7.sgy`
- 曲率体：
  `/data/shared/project-oil/wx数据/砂砾岩/补充材料-20260623/车西-最大曲率T4-T7.sgy`
  `/data/shared/project-oil/wx数据/砂砾岩/补充材料-20260623/车西_T4-T7/车西-最大正曲率T4-T7.sgy`
- 层位：
  `/data/shared/project-oil/wx数据/砂砾岩/层位`
- 原始断层解释：
  `/data/shared/project-oil/wx数据/砂砾岩/断层/FaultStick-GeoEast.dat`

## Step6B/6C/6D：多尺度地震解释证据体

### Step6A 小尺度背景密度

实现口径：

- 不重新训练模型。
- 直接使用当前 Step6 的三维密度体作为小尺度背景裂缝密度。
- 小尺度结果只表示测井尺度裂缝发育概率，不强行解释为断层或中尺度裂缝带。

输出：

- `small_density.sgy`
- `small_prior.sgy`

检查点：

- trace 数、采样轴、trace header 与原始 Step6B 输入保持一致。
- 小尺度密度范围不被 Step6B/6C 的地震属性异常强行放大。

### Step6B 中尺度裂缝带证据体

实现口径：

- 中尺度以蚂蚁体高值为主，因为蚂蚁体更接近裂缝/小断裂的体属性表达。
- 相干体低值和曲率高值只作为辅助增强，不作为唯一判据。
- 去除孤立噪声体素，保留连续的带状/片状发育区。
- 不再把中尺度裂缝带抽象成一条中心线，后续 Step7B 直接对候选带体素进行局部面片化。

计算逻辑：

- `AntTrackScore = high_score(AntTrack)`
- `LowCoherenceScore = low_score(Coherence)`
- `CurvatureScore = max(high_score(CurvatureMax), high_score(CurvaturePos))`
- `MediumPrior = AntTrackScore * (0.65 + 0.20 * LowCoherenceScore + 0.15 * CurvatureScore)`
- 对 `MediumPrior` 做三维连通域过滤，删除小孤立体。

输出：

- `medium_prior.sgy`
- `medium_candidate_mask.sgy`
- `multiscale_density_bundle_summary.json`

检查点：

- 中尺度高值区域应明显贴近蚂蚁体高值。
- 连通域数量不能过多，不能退化成满屏孤立小片。
- 中尺度候选体的局部 PCA 倾向倾角应有空间变化，不应固定成同质化模板。

### Step6C 大尺度断层/断裂带证据体

实现口径：

- 原始断层解释是硬约束，应保留真实三维曲面形态。
- 低相干只用于补充未解释小断层/断裂带，但必须过滤横向层间黑条带。
- 大尺度候选重点关注陡倾、垂向连续、空间上成带的低相干异常。

计算逻辑：

- 从相干体低值中提取 `LowCoherenceScore`。
- 对低相干候选做连通域分析。
- 对每个连通域计算水平跨度、时间跨度、局部 PCA 法向/倾角。
- 横向薄层状异常降权或剔除；垂向连续、陡倾异常保留。
- 原始 `FaultStick-GeoEast.dat` 后续在 Step7C 直接转成断层曲面，不只作为密度增强。

输出：

- `large_prior.sgy`
- `large_candidate_mask.sgy`
- `large_component_summary.csv`

检查点：

- 大尺度候选不应主要由水平层位界面构成。
- 候选异常应与原始断层或相干体竖向黑灰色不连续带更接近。

### Step6D 最终小尺度衍生密度修正

实现口径：

- Step6D 不再把 `small/medium/large` 简单相加成一个总密度体。
- Step6D 读取 Step6A 原始小尺度背景密度，并根据 Step6B 中尺度裂缝带、Step6C 大尺度断层周边损伤带增强小尺度密度。
- 输出的 `final_small_density` 只表示小尺度/衍生小裂缝密度，不包含中尺度裂缝带或大尺度断层本体。
- Step7A 读取 `final_small_density`，Step7B 读取 Step6B，Step7C 读取 Step6C。

输出：

- `final_small_density.sgy`
- `final_small_candidate_mask.sgy`
- `multiscale_prior_bundle.npz`
- `multiscale_density_bundle_summary.json`

检查点：

- `npz` 中至少包含：`base_small_density`、`final_small_density`、`medium_damage`、`large_damage`、`medium_prior`、`medium_mask`、`large_prior`、`large_mask`、`x_values`、`y_values`、`samples`。
- 中/大尺度核心区不能被 Step6D 直接改造成小尺度高密度实心块；增强应主要表达核心外侧损伤带。

## Step7A 小尺度 DFN

实现口径：

- 基于当前旧 Step7/Step7B 中的小尺度裂缝片生成逻辑改造。
- 小尺度裂缝片数量由 Step6A 密度控制。
- 倾向倾角优先由局部三维密度 PCA 估计；如果局部点不足，再使用层段默认方向。
- 裂缝片大小由局部密度控制，但尺度应明显小于中尺度和大尺度。

输出：

- `step7a_small_scale_dfn/output/candidate_cheye1_multiscale_v1/small_dfn_patches.csv`
- `small_dfn_raw_time.vtk`

检查点：

- 小尺度片不应全部水平或全部同一倾角。
- 小尺度片不能充满中/大尺度核心区，以免视觉上覆盖主要构造。

## Step7B 中尺度裂缝带 DFN

实现口径：

- 不再使用“中心线串片”作为主逻辑。
- 对 Step6B 的中尺度候选带做局部三维窗口 PCA，直接从候选带体素云估计局部裂缝面方向。
- 裂缝片中心沿候选带内部的高分体素采样，而不是沿抽象中心线等距采样。
- 裂缝片大小随局部带宽、局部厚度、局部 prior 强度变化。
- 中尺度片之间通过较小间距和重叠率体现连续性，但不牺牲局部形态。

核心几何规则：

- 倾向/倾角：由局部候选带体素 PCA 的平面法向计算。
- 片长：与局部带宽、局部平面主轴跨度、prior 强度正相关。
- 片高：与局部时间厚度、prior 强度正相关。
- 中心位置：候选带内部高 prior 体素，允许在同一连通带内空间变化。
- 形状：本版先输出矩形面片，但保留 `LocalBandWidthM`、`LocalBandThicknessMs`、`PatchAreaM2` 等字段，后续可替换为多边形片。

输出：

- `step7b_multiscale_initial_dfn/output/candidate_cheye1_multiscale_v4/medium_dfn_patches.csv`
- `medium_dfn_raw_time.vtk`
- `medium_candidate_components_raw_time.vtk`
- `medium_generation_audit.csv`

检查点：

- `AzimuthDeg/DipDeg` 必须来自局部带 PCA，不能是预设常数。
- `PatchAreaM2/LengthM/HeightTimeMs` 应有明显差异。
- 中尺度片要与蚂蚁体高值带、相干体局部不连续带保持空间一致。

## Step7C 大尺度断层/断裂带 DFN

实现口径：

- 原始断层解释成果作为硬约束，优先生成与原始单元断层面类似的三维曲面。
- 不把断层曲面压缩成小矩形碎片链条。
- 对原始单元断层 patch 按 demo 区域裁剪，并直接保留 raw surface fragments 的顶点几何。
- Regional fault panels 只作为断层影响带构造依据，不再替代原始断层面本体。
- 对未解释但 Step6C 高置信的大尺度低相干异常，可作为补充断裂带面片组，但需要明确 `SourceType=large_lowcoh_inferred_panel`。

核心几何规则：

- 原始断层曲面：调用旧断层后融合工具中的 `run_build_fault_surface_fragments()`，直接读取单元断层 patch 的 surface fragments 和 `V1X..V4Z` 顶点。
- 断层影响带：调用 `run_build_regional_fault_panels()` 生成 regional panels，再沿 panel 法向偏移生成 damage-zone patch，方向和大小继承 regional panel。
- 大尺度低相干补充：读取 Step6C `large_fault_prior_components.npz`，在每个陡倾、垂向连续 component 内沿主方向拆成多个连续 panel，而不是一个 component 一个大矩形。

输出：

- `step7c_large_fault_dfn/output/candidate_cheye1_step7c_large_from_step6c_rebuild_v3_surface_panelgroup_preview/large_fault_surface_raw_time.vtk`
- `large_fault_only_and_influence_raw_time.vtk`
- `large_lowcoh_component_panels_raw_time.vtk`
- `large_fault_and_damage_raw_time.vtk`
- `large_fault_and_damage_patches.csv`
- `large_generation_audit.csv`

检查点：

- 原始断层曲面方向应与 `FaultStick-GeoEast.dat` 在 XY/T 空间一致。
- 断层曲面不能因过度细化丢失主连通性。
- 大尺度片的倾向倾角应由断层面/候选体局部几何计算。

## Step7D 多尺度融合

实现口径：

- 合并 Step7A/7B/7C 输出。
- 大尺度断层为硬约束，中尺度次之，小尺度为背景。
- 融合时只做必要去重和裁剪，不把大中尺度重新打散。
- 近断层区域的小尺度片可适当降采样，避免大尺度断层附近视觉过密。

输出：

- `step7d_multiscale_fused_dfn/output/candidate_cheye1_multiscale_v1/fused_multiscale_dfn_patches.csv`
- `fused_multiscale_dfn_raw_time.vtk`
- `fused_multiscale_summary.json`

检查点：

- 输出字段统一，至少包含 `PatchID`、`FractureScale`、`SourceType`、`CenterX`、`CenterY`、`CenterTime`、`LengthM`、`HeightTimeMs`、`AzimuthDeg`、`DipDeg`、`PatchAreaM2`、`SourceDensity`、`Confidence`。
- VTK 可用 `FractureScaleCode`、`PatchAreaM2`、`SourceDensityRenderNorm` 分类渲染。

## Step8 测井纠偏

实现口径：

- 输入改为 Step7D 融合 DFN。
- 成像测井真实裂缝仍作为硬约束，但新增/修正的井控片尺度不能压过大中尺度地质约束。
- 常规测井预测裂缝只作为局部补充和纠偏，不改变断层曲面主体。
- 保留中间 VTK，便于定位问题。

输出：

- `step8_dfn_well_correction/output/candidate_cheye1_multiscale_v1/well_corrected_dfn_fracture_patches.csv`
- `well_corrected_dfn_raw_time.vtk`
- `00_step7d_initial_input_all_raw_time.vtk`
- `05_step8_added_well_control_only_raw_time.vtk`

检查点：

- 井控裂缝中心不能全部机械地压回井轨迹；允许小范围偏移。
- 成像测井真实裂缝的倾向倾角必须来自 Step3 真实解释字段。
- 不能出现异常超长裂缝片。

## Step9 剖面图

实现口径：

- 输入改为 Step8 修正后的融合 DFN。
- 剖面图显示 DFN 与相干体、蚂蚁体、原始成像测井裂缝解释、原始断层解释的匹配程度。
- XZ/YZ 都画裂缝片与剖面的交线，而不是只画中心点。
- 原始断层叠加使用 `FaultStick-GeoEast.dat` 的原始轨迹/曲面与剖面交线，不使用后处理后的断层裂缝片冒充原始解释。

输出：

- `step9_section_visualize/output/candidate_cheye1_multiscale_v1/cheye1_dfn_coherence_sections/`
- 保留最终 8 张核心图：
  - XZ DFN
  - YZ DFN
  - XZ coherence
  - YZ coherence
  - XZ coherence + DFN
  - YZ coherence + DFN
  - XZ well-near 200 m coherence + DFN
  - YZ well-near 200 m coherence + DFN

检查点：

- 大尺度断层在剖面上应与原始断层解释交线一致。
- 中尺度裂缝应更贴近蚂蚁体高值/相干体局部断续黑灰带。
- 小尺度裂缝只作为背景，不应遮盖大中尺度主体。

## 执行顺序

1. 新增并运行 Step6 多尺度证据体脚本，只做 `candidate_cheye1`，输出 `candidate_cheye1_multiscale_v1`。
2. 实现并运行 Step7A 小尺度 DFN。
3. 修改新 `step7b_multiscale_initial_dfn`，实现中尺度候选带局部面片化，输出 `candidate_cheye1_multiscale_v4`。
4. 新增并运行 Step7C 大尺度断层曲面和影响带生成。
5. 新增并运行 Step7D 多尺度融合。
6. 新增 Step8 配置，基于 Step7D 输出做测井纠偏。
7. 新增 Step9 配置，基于 Step8 输出生成最终剖面图。
8. 汇报每一步输出目录、patch 数量、尺度分布和最终图片目录。

## 当前风险

- 如果 Step6C 仍把水平层位低相干当作大尺度断层，大尺度 DFN 会继续偏水平，需要在 Step6C 连通域过滤中处理。
- 如果中尺度只按中心线采样，会丢失真实带状/面状信息，因此 Step7B 必须从候选带体素局部 PCA 直接生成片。
- 如果原始断层 stick 点的断层编号/分组字段不稳定，需要先自动识别列名，并输出原始断层裁剪 QC。
- Step8 不能再把融合后的大中尺度片整体放大或重新解释，否则会破坏 Step7D 结果。
