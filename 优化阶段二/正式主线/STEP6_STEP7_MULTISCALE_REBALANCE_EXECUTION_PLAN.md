# Step6-Step7 多尺度重构再平衡执行方案

## 1. 本轮目标

本轮不再继续在单一裂缝密度体上微调权重，而是把裂缝发育信息按尺度拆开处理，再进入 DFN 构造：

- Step6A：小尺度背景裂缝密度，对应测井尺度裂缝和不同地层内的背景裂缝。
- Step6B：中尺度裂缝带先验，对应蚂蚁体高值、局部低相干和曲率异常共同支持的裂缝带/小断层组合。
- Step6C：大尺度断层和断裂带先验，对应原始断层解释成果、相干体陡向低值不连续带，以及未解释的小断层候选。
- Step6D：最终小尺度衍生密度修正，读取 Step6A/B/C，将大中尺度构造的损伤带影响加入小尺度背景密度；输出给 Step7A 使用的最终小尺度密度，同时保留中/大尺度索引用于 QC。
- Step7A/B/C/D：按小、中、大尺度分别生成 DFN，再融合；倾向、倾角、片大小和位置都由对应尺度的真实几何证据控制。
- Step8/Step9：在融合 DFN 基础上做测井纠偏和剖面图验证，重点说明 DFN 与相干体、蚂蚁体、原始成像测井裂缝、原始断层解释的一致性。

执行原则：先做 Step6 证据体和 QC，再做 Step7 DFN；每个阶段使用新输出目录，不覆盖已有结果。

## 2. 当前问题和根因

### 2.1 Step7A 小尺度裂缝太稀疏

当前小尺度裂缝片数量偏少，背景裂缝表达不足。原因主要是当前 Step7A 使用的候选分位数和采样强度比旧版 Step7B 严格很多，导致小尺度只保留了高密度核心点，没有形成足够的背景裂缝。

调整方向：

- 小尺度可以复用旧 Step7B 更宽松的候选策略，但只服务于背景裂缝，不参与大中尺度断裂带表达。
- 小尺度片尺寸必须小于中尺度和大尺度，避免视觉上抢占主体。
- 小尺度倾向倾角优先由局部三维密度梯度/PCA 估计，局部信息不足时才使用层段默认方向。

### 2.2 Step7B 中尺度裂缝片过大、过密、连续性不清楚

当前中尺度结果容易被少数大 patch 主导，视觉上不像沿裂缝带连续发育，而像在候选区域内铺了很多大矩形片。根因是 Step6B 中尺度候选区域存在大连通体，Step7B 又按局部窗口面片化，缺少带宽、方向稳定性、片间距和重叠率的联合控制。

调整方向：

- Step6B 先把中尺度候选带切成局部可解释的 corridor/component，不能让一个大连通体覆盖整个 demo 区。
- Step7B 不使用抽象中心线作为唯一依据，但需要在候选带内部建立局部走向连续约束。
- 中尺度 patch 大小随局部带宽、厚度和 prior 强度变化，但设置上限，避免全部顶到最大尺寸。
- 中尺度 patch 间距应小于片长，允许适度重叠，让人能看出连续裂缝带。

### 2.3 Step7C 大尺度断层与真实断层对不齐、面积偏小

当前 Step7C 主要读取原始 `FaultStick-GeoEast.dat` 在 demo 内的点，demo 内点数有限，导致原始断层约束不足；同时未充分利用已经完成的单元断层面切割结果。

调整方向：

- 原始大断层优先从单元分割后的 fault patch/panel 读取，而不是只从点云重新拟合。
- `FaultStick-GeoEast.dat` 仍作为原始断层轨迹和时间域坐标的权威来源，用于校验 patch/panel 是否对齐。
- Step6C 还要识别未解释的小断层候选，不能只依赖现有断层解释成果。
- 大尺度输出应是连续或半连续曲面/面片组，不能退化成少量碎小矩形片。

### 2.4 Step6 输出只能部分可信

当前 Step6 原始三维密度体更适合作为小尺度背景裂缝概率，不能直接解释所有大中尺度裂缝。相干体和蚂蚁体表达的是地震不连续和裂缝带响应，如果不在 Step6 阶段拆分尺度，后续 Step7 很难同时满足井点、地震属性和断层解释三类约束。

调整方向：

- Step6A 保留当前三维密度体作为小尺度背景的起点，先不重训。
- Step6B/6C 单独从地震解释属性构造中/大尺度先验。
- Step6D 输出多尺度 bundle，让 Step7 知道每个候选位置的尺度和证据来源。

## 3. 版本和目录规则

本轮新结果统一使用后缀：

- `candidate_cheye1_multiscale_rebalance_v1`

不覆盖以下已有目录：

- `candidate_cheye1_multiscale_v1`
- `candidate_cheye1_multiscale_v4`
- `candidate_cheye1_multiscale_preview_v1`
- `candidate_cheye1_multiscale_preview_v2`
- `candidate_cheye1_multiscale_preview_v3`
- `candidate_cheye1_detail_preserve_v4_fault_postfusion_v2`
- `candidate_cheye1_fused_lowcoh_steep`

建议新增输出目录：

- Step6A：`step6b_demo_density_volume_3d/output/candidate_cheye1_multiscale_rebalance_v1/step6a_small/`
- Step6B：`step6b_demo_density_volume_3d/output/candidate_cheye1_multiscale_rebalance_v1/step6b_medium/`
- Step6C：`step6b_demo_density_volume_3d/output/candidate_cheye1_multiscale_rebalance_v1/step6c_large/`
- Step6D：`step6b_demo_density_volume_3d/output/candidate_cheye1_multiscale_rebalance_v1/step6d_bundle/`
- Step7A：`step7a_small_scale_dfn/output/candidate_cheye1_multiscale_rebalance_v1/`
- Step7B：`step7b_multiscale_initial_dfn/output/candidate_cheye1_multiscale_rebalance_v1/`
- Step7C：`step7c_large_fault_dfn/output/candidate_cheye1_multiscale_rebalance_v1/`
- Step7D：`step7d_multiscale_fused_dfn/output/candidate_cheye1_multiscale_rebalance_v1/`
- Step8：`step8_dfn_well_correction/output/candidate_cheye1_multiscale_rebalance_v1/`
- Step9：`step9_section_visualize/output/candidate_cheye1_multiscale_rebalance_v1/cheye1_dfn_coherence_sections/`

## 4. 阶段 0：数据和现有结果复核

目标：先确认输入数据、坐标、时间轴和现有输出能否支撑重构，不直接生成新 DFN。

主要输入：

- 当前 Step6 三维密度体：`step6b_demo_density_volume_3d/output/candidate_cheye1/candidate_cheye1_3d_predicted_density.sgy`
- 当前 trace mapping：`step6b_demo_density_volume_3d/output/candidate_cheye1/candidate_cheye1_3d_trace_mapping.npz`
- 相干体、蚂蚁体、曲率体 T4-T7 数据。
- 原始断层：`/data/shared/project-oil/wx数据/砂砾岩/断层/FaultStick-GeoEast.dat`
- 单元断层分割结果：`小范围DFN生成/断层裂缝片生成/断层划分切割/fault_patches_out/patches`
- 单元断层汇总：`小范围DFN生成/断层裂缝片生成/断层划分切割/fault_patches_out/fault_patches_summary.csv`

执行内容：

- 检查三维密度体与 trace mapping 的 trace 数、XY、sample/time 是否一致。
- 检查相干体、蚂蚁体、曲率体能否按 demo 区域 trace header 对齐。
- 检查单元断层 patch 的 XY/T 范围是否覆盖 `candidate_cheye1` demo 区域。
- 检查 `FaultStick-GeoEast.dat` 与单元断层 patch 是否在同一 XY/T 坐标系。

建议脚本：

- 新增或扩展：`step6b_demo_density_volume_3d/qc_multiscale_rebalance_inputs.py`

输出：

- `step6b_demo_density_volume_3d/output/candidate_cheye1_multiscale_rebalance_v1/input_qc/input_qc_summary.json`
- `step6b_demo_density_volume_3d/output/candidate_cheye1_multiscale_rebalance_v1/input_qc/fault_patch_demo_overlap.csv`
- `step6b_demo_density_volume_3d/output/candidate_cheye1_multiscale_rebalance_v1/input_qc/attribute_alignment_qc.csv`

通过标准：

- 三维密度体、地震属性体和 trace mapping 可在同一 XY/T 网格上采样。
- 原始断层和单元断层 patch 的时间坐标确认是 ms，不混入深度域。
- 如果单元断层 patch 覆盖 demo，则 Step7C 必须优先使用 patch/panel；如果不覆盖，则 Step7C 退回原始 fault stick + 自识别断层候选。

不通过处理：

- 坐标不一致时停止，不进入 Step6B/6C。
- 断层 patch 与 FaultStick 时间域不一致时，不复用旧 patch，改为基于 `FaultStick-GeoEast.dat` 重新构面。

## 5. 阶段 1：Step6A 小尺度背景密度整理

目标：把当前三维密度体整理成小尺度背景裂缝通道，增加 QC，不重新训练模型。

主要输入：

- 当前 Step6 三维密度体 SGY。
- trace mapping NPZ。
- 层位 T4-T7。
- 当前测井/成像测井裂缝密度标签 QC 结果。

执行内容：

- 读取当前三维密度体，裁剪到 `candidate_cheye1` demo 区域和 T4-T7。
- 做稳健归一化，避免极端值主导采样。
- 输出小尺度密度和小尺度候选 mask。
- 统计小尺度高值与井点真实/预测裂缝密度的关系。

建议脚本：

- 新增：`step6b_demo_density_volume_3d/build_step6a_small_background.py`

输出：

- `step6a_small/small_background_density.sgy`
- `step6a_small/small_background_prior.npz`
- `step6a_small/small_background_qc.json`

关键参数初值：

- `density_clip_quantile_low = 0.02`
- `density_clip_quantile_high = 0.995`
- `small_candidate_quantile = 0.88`
- `small_core_quantile = 0.95`

通过标准：

- 小尺度候选体素占比不能过低，避免背景裂缝稀疏。
- 小尺度 high-density 区域不应只集中在大断层附近。
- 输出仍使用原始 trace header，方便后续写 SGY 和对齐剖面。

不通过处理：

- 如果小尺度过稀疏，降低 `small_candidate_quantile` 到 `0.84-0.86`。
- 如果小尺度过密，增加最小距离采样或分层采样，而不是提高到只剩核心高值。

## 6. 阶段 2：Step6B 中尺度裂缝带先验重构

目标：以蚂蚁体高值为主，识别中尺度裂缝带、小断层组合和损伤带，不再让一个大连通体控制全区。

主要输入：

- 蚂蚁体 T4-T7。
- 相干体 T4-T7。
- 最大曲率/最大正曲率 T4-T7。
- Step6A 小尺度背景密度。
- 原始断层距离场或 Step6C 初步大断层 mask。

执行内容：

- 对蚂蚁体做高值评分 `AntTrackScore`。
- 对相干体做低值评分 `LowCoherenceScore`，但只作为辅助，不让水平层界面直接进入中尺度。
- 对曲率体做高值评分 `CurvatureScore`。
- 构造中尺度 prior：蚂蚁体为主，低相干和曲率为辅助。
- 对中尺度候选做三维连通域分割，进一步按局部方向、空间间断和薄层特征切分大连通体。
- 输出每个中尺度 component 的局部 PCA 方向、带宽、厚度、长度、平均 prior。

建议脚本：

- 新增：`step6b_demo_density_volume_3d/build_step6b_medium_corridor_prior.py`

计算口径：

- `MediumScore = AntTrackScore * (0.55 + 0.25 * LowCoherenceScore + 0.20 * CurvatureScore)`
- 孤立噪声要求：候选体素需要满足最小邻域支持。
- 水平层界面过滤：横向跨度大、时间厚度小、局部法向接近垂直的异常降权。
- 大连通体切分：按 XY 网格块、局部 PCA 方向突变和 prior 低谷切分，而不是只取一个 component。

输出：

- `step6b_medium/medium_corridor_prior.sgy`
- `step6b_medium/medium_corridor_mask.sgy`
- `step6b_medium/medium_corridor_components.npz`
- `step6b_medium/medium_corridor_components_raw_time.vtk`
- `step6b_medium/medium_corridor_component_summary.csv`
- `step6b_medium/medium_corridor_qc.json`

关键参数初值：

- `anttrack_high_quantile = 0.82`
- `medium_candidate_quantile = 0.88`
- `min_component_voxels = 80`
- `max_component_voxels_before_split = 25000`
- `min_vertical_extent_ms = 12`
- `max_horizontal_layer_thickness_ms = 8`

通过标准：

- 中尺度 prior 高值应明显贴近蚂蚁体高值。
- 最大 component 体素数不能超过中尺度候选总体素数的 40%。
- 输出的 component 在 ParaView 中应表现为多个带状/片状区域，而不是一个覆盖全区的大块。
- component 的局部方向、厚度、带宽有统计差异，不能全部同质化。

不通过处理：

- 如果最大 component 仍过大，优先加强 component 切分，不直接降低 prior。
- 如果中尺度贴不住蚂蚁体，调整 AntTrackScore 权重，不用相干体强行补。
- 如果水平层界面误入过多，加强薄层过滤和陡倾/局部方向筛选。

## 7. 阶段 3：Step6C 大尺度断层/断裂带先验重构

目标：大尺度部分同时保留地质人员解释的原始断层和自行识别的未解释小断层候选。

主要输入：

- 原始断层 `FaultStick-GeoEast.dat`。
- 单元分割断层 patch/panel 目录。
- 相干体低值异常。
- 蚂蚁体高值辅助。
- 曲率异常辅助。
- demo 区域边界和 T4-T7。

执行内容：

- 优先从单元分割 fault patch/panel 中筛选与 demo 区域相交的原始断层面。
- 读取 `FaultStick-GeoEast.dat`，用于原始轨迹校验和剖面图 overlay。
- 根据相干体低值提取未解释断层候选。
- 过滤横向层间不连续：横向连续、时间厚度薄、缺少垂向连续性的低相干条带不作为大断层。
- 对保留的大尺度候选计算局部曲面方向、倾角、面积和可信度。

建议脚本：

- 新增或重构：`step6b_demo_density_volume_3d/build_step6c_large_fault_prior.py`

输出：

- `step6c_large/large_fault_prior.sgy`
- `step6c_large/large_fault_mask.sgy`
- `step6c_large/original_fault_panels_selected_raw_time.vtk`
- `step6c_large/inferred_large_fault_candidates_raw_time.vtk`
- `step6c_large/large_fault_component_summary.csv`
- `step6c_large/large_fault_qc.json`

关键参数初值：

- `coherence_low_quantile = 0.20`
- `large_candidate_quantile = 0.90`
- `min_vertical_extent_ms = 24`
- `min_dip_deg = 45`
- `min_component_voxels = 120`
- `fault_patch_margin_m = 500`

通过标准：

- 原始断层 patch/panel 如果覆盖 demo，必须在输出中可见且方向保持。
- 未解释断层候选应主要对应相干体竖向黑灰色不连续带，而不是水平黑条带。
- 大尺度候选数量不能只剩原始断层，也不能满屏都是低相干噪声。

不通过处理：

- 如果原始断层 patch 与 demo 不相交，记录原因，并切换到 `FaultStick-GeoEast.dat` 构面。
- 如果 inferred fault 大量水平，增加倾角和垂向连续门槛。
- 如果 inferred fault 过少，降低 `coherence_low_quantile` 的严格程度，但必须保留水平层界面过滤。

## 8. 阶段 4：Step6D 最终小尺度衍生密度修正

目标：把 Step6A 的原始小尺度背景密度修正为最终小尺度裂缝密度。Step6D 只把 Step6B/Step6C 的中大尺度构造作为损伤带来源，不把中大尺度裂缝本体混入小尺度结果。

主要输入：

- Step6A 原始小尺度背景密度。
- Step6B 中尺度 prior、mask、component summary。
- Step6C 大尺度 prior、mask、fault panels 和 inferred components。

建议脚本：

- 新增：`step6b_demo_density_volume_3d/build_step6d_multiscale_bundle.py`

输出：

- `step6d_bundle/multiscale_prior_bundle.npz`
- `step6d_bundle/final_small_density.sgy`
- `step6d_bundle/final_small_candidate_mask.sgy`
- `step6d_bundle/scale_label.sgy`
- `step6d_bundle/scale_confidence.sgy`
- `step6d_bundle/multiscale_bundle_summary.json`

bundle 必须包含字段：

- `base_small_density`
- `base_small_mask`
- `final_small_density`
- `final_small_mask`
- `medium_damage`
- `large_damage`
- `medium_prior`
- `medium_mask`
- `medium_component_id`
- `medium_component_features`
- `large_prior`
- `large_mask`
- `large_component_id`
- `large_component_features`
- `x_values`
- `y_values`
- `sample_times_ms`
- `trace_headers`

通过标准：

- Step7A 只读取 `final_small_density` 和 `final_small_mask`，不直接读取 Step6B/Step6C 来生成小裂缝。
- Step7B/Step7C 分别读取 Step6B/Step6C，不从 Step6D 的小尺度密度反推中大尺度裂缝。
- `scale_label` 中 large/medium/small 有明确优先级：large > medium > small。
- Step6D 输出的最终密度只表示小尺度/衍生小裂缝密度，不表示大中尺度裂缝本体。

不通过处理：

- 如果中大尺度核心区小尺度过强，增强只保留在损伤带外围，核心区做降权或限幅。
- 如果远离构造区小尺度被整体抬升，降低 damage 扩散半径或增强系数。

## 9. 阶段 5：Step7A 小尺度 DFN 再平衡

目标：增加背景裂缝表达，但不让小尺度淹没大中尺度。

主要输入：

- Step6D bundle 中的 `final_small_density`、`final_small_mask`。
- 当前旧 Step7/旧 Step7B 的采样参数作为参考。

建议脚本：

- 修改或新增配置运行：`step7a_small_scale_dfn/build_small_scale_dfn.py`

输出：

- `step7a_small_scale_dfn/output/candidate_cheye1_multiscale_rebalance_v1/small_dfn_patches.csv`
- `step7a_small_scale_dfn/output/candidate_cheye1_multiscale_rebalance_v1/small_dfn_raw_time.vtk`
- `step7a_small_scale_dfn/output/candidate_cheye1_multiscale_rebalance_v1/small_generation_audit.csv`

关键参数初值：

- `candidate_density_quantile = 0.88`
- `core_density_quantile = 0.95`
- `target_patch_count = 3500-6000`
- `min_patch_count = 2500`
- `max_patch_count = 9000`
- `length_range_m = 18-75`
- `height_range_ms = 4-18`

几何规则：

- 中心位置从小尺度高密度体素中分层采样。
- 倾向/倾角优先使用局部小尺度密度 PCA 或邻域梯度。
- 片长和片高随局部密度变化，但上限严格小于中尺度。
- 位于 large hard fault 核心区的小尺度片只降采样，不全部删除。

通过标准：

- patch 数量明显高于当前 936/1474 级别，能表达背景裂缝。
- 小尺度 patch 面积统计与中尺度、大尺度明显分离。
- VTK 中小尺度不是全部水平片，也不是全部同一倾角。

不通过处理：

- 如果小尺度仍太少，先降候选分位数，再提高 count_scale。
- 如果小尺度太乱，增加空间最小间距和分层采样。

## 10. 阶段 6：Step7B 中尺度连续裂缝带 DFN 再平衡

目标：中尺度裂缝要沿 Step6B corridor 成带、成组、连续，同时片大小不能压过大尺度断层。

主要输入：

- Step6D bundle 中的 `medium_prior`、`medium_mask`、`medium_component_id`、`medium_component_features`。

建议脚本：

- 修改：`step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py`
- 新配置：`step7b_multiscale_initial_dfn/configs/formal_candidate_cheye1_step7b_medium_rebalance_v1.json`

输出：

- `step7b_multiscale_initial_dfn/output/candidate_cheye1_multiscale_rebalance_v1/medium_dfn_patches.csv`
- `step7b_multiscale_initial_dfn/output/candidate_cheye1_multiscale_rebalance_v1/medium_dfn_raw_time.vtk`
- `step7b_multiscale_initial_dfn/output/candidate_cheye1_multiscale_rebalance_v1/medium_corridor_components_raw_time.vtk`
- `step7b_multiscale_initial_dfn/output/candidate_cheye1_multiscale_rebalance_v1/medium_generation_audit.csv`

关键参数初值：

- `target_patch_count = 700-1300`
- `length_range_m = 90-180`
- `height_range_ms = 14-32`
- `max_patch_area = 5400`
- `spacing_to_length_ratio = 0.45-0.65`
- `min_overlap_ratio_along_band = 0.20`

几何规则：

- patch 中心从 medium corridor 内部高分体素采样，不从单条抽象中心线等距生成。
- 倾向/倾角由局部 corridor 体素 PCA 计算，邻近 patch 方向做平滑约束。
- 片长由局部带宽和 component 主方向长度控制；片高由局部时间厚度控制。
- 同一 component 内 patch 间距小于片长，保证视觉连续性。
- 形状本轮仍可使用矩形片，但必须输出 `LocalBandWidthM`、`LocalBandThicknessMs`、`PatchAreaM2`、`ComponentID`。

通过标准：

- 中尺度 patch 不再大量顶到最大尺寸。
- 中尺度 patch 与 `medium_corridor_components_raw_time.vtk` 在空间上重合。
- 相邻 patch 有方向连续性，能看出裂缝带走向。
- 中尺度面积中位数应低于大尺度，高于小尺度。

不通过处理：

- 如果中尺度仍显得过大，优先降低 length/height 上限。
- 如果连续性不明显，降低 `spacing_to_length_ratio` 或增加同 component 内 patch 数。
- 如果方向同质化，检查 PCA 邻域是否过大或 component 切分是否过粗。

## 11. 阶段 7：Step7C 大尺度断层/断裂带 DFN 再平衡

目标：大尺度结果应接近原始断层曲面，同时补充未解释小断层，不再只输出少量小片。

主要输入：

- Step6C 输出的原始断层 panels 和 inferred large fault candidates。
- 单元分割 fault patch/panel。
- `FaultStick-GeoEast.dat` 原始轨迹。

建议脚本：

- 修改：`step7c_large_fault_dfn/build_large_fault_dfn.py`
- 新配置：`step7c_large_fault_dfn/configs/formal_candidate_cheye1_step7c_large_rebalance_v1.json`

输出：

- `step7c_large_fault_dfn/output/candidate_cheye1_multiscale_rebalance_v1/original_fault_surface_raw_time.vtk`
- `step7c_large_fault_dfn/output/candidate_cheye1_multiscale_rebalance_v1/inferred_large_fault_surface_raw_time.vtk`
- `step7c_large_fault_dfn/output/candidate_cheye1_multiscale_rebalance_v1/large_fault_and_damage_patches.csv`
- `step7c_large_fault_dfn/output/candidate_cheye1_multiscale_rebalance_v1/large_fault_and_damage_raw_time.vtk`
- `step7c_large_fault_dfn/output/candidate_cheye1_multiscale_rebalance_v1/large_generation_audit.csv`

几何规则：

- 原始断层：优先复用单元分割 patch/panel 的多边形曲面，不重新压缩成小矩形。
- 原始断层方向：由 patch/panel 局部法向计算，必须与原始曲面一致。
- 断层影响带：在原始断层附近生成少量平行或近似平行诱导片，尺寸小于主断层面。
- 自识别断层：只使用 Step6C 中陡倾、垂向连续的大尺度候选，方向由候选体局部 PCA 计算。

关键参数初值：

- `fault_patch_margin_m = 500`
- `damage_zone_width_m = 100-250`
- `main_fault_min_area = 10000`
- `inferred_fault_length_range_m = 220-500`
- `inferred_fault_height_range_ms = 35-90`
- `min_dip_deg = 45`

通过标准：

- 原始断层 surface 在 ParaView 中与原始 patch/panel 方向一致。
- 大尺度主断层面积显著大于中尺度 patch。
- 自识别断层数量有限，主要对应相干体竖向低值带。
- 输出中清楚区分 `SourceType=original_fault_panel`、`fault_damage_zone`、`large_lowcoh_inferred`。

不通过处理：

- 如果原始断层对不上，停止 Step7C，回查 Step6C patch 读取和坐标轴。
- 如果大尺度面积过小，优先保留原始 panel 多边形，不再用固定矩形替代。
- 如果自识别断层太乱，先只输出原始断层和 damage zone，再调整 Step6C。

## 12. 阶段 8：Step7D 多尺度融合再平衡

目标：融合小、中、大尺度 DFN，保留尺度差异，不让任一尺度视觉上完全压制其他尺度。

主要输入：

- Step7A `small_dfn_patches.csv`
- Step7B `medium_dfn_patches.csv`
- Step7C `large_fault_and_damage_patches.csv`

建议脚本：

- 修改：`step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py`
- 新配置：`step7d_multiscale_fused_dfn/configs/formal_candidate_cheye1_step7d_fused_rebalance_v1.json`

输出：

- `step7d_multiscale_fused_dfn/output/candidate_cheye1_multiscale_rebalance_v1/fused_multiscale_dfn_patches.csv`
- `step7d_multiscale_fused_dfn/output/candidate_cheye1_multiscale_rebalance_v1/fused_multiscale_dfn_raw_time.vtk`
- `step7d_multiscale_fused_dfn/output/candidate_cheye1_multiscale_rebalance_v1/fused_multiscale_summary.json`
- `step7d_multiscale_fused_dfn/output/candidate_cheye1_multiscale_rebalance_v1/fused_scale_qc.csv`

融合规则：

- 大尺度为硬约束，不被中小尺度删除。
- 中尺度在大断层核心区可适度降采样，但不能被完全删除。
- 小尺度在大中尺度核心区降采样，保留背景，不让视觉过密。
- 每个 patch 必须保留 `FractureScale`、`SourceType`、`PatchAreaM2`、`SourceDensity`、`Confidence`。

通过标准：

- 小、中、大尺度数量和面积分布有明显区分。
- `PatchAreaM2` 不出现异常极大值或全尺度相同值。
- VTK 可按 `FractureScaleCode`、`SourceTypeCode`、`PatchAreaM2` 渲染。

不通过处理：

- 如果小尺度太少，放宽 near-structure downsample。
- 如果中尺度主导画面，降低中尺度 patch 上限或数量。
- 如果大尺度不明显，优先修 Step7C，不在 Step7D 人工放大。

## 13. 阶段 9：Step8 测井纠偏

目标：在不破坏大中尺度地震解释约束的前提下，让井旁 DFN 与真实成像测井和常规测井预测保持一致。

主要输入：

- Step7D 融合 DFN。
- Step3 成像测井真实裂缝解释。
- 常规测井预测裂缝点位/密度。
- 井轨迹。

建议脚本：

- 修改现有：`step8_dfn_well_correction`
- 新配置：`formal_candidate_cheye1_step8_multiscale_rebalance_v1.json`

输出：

- `step8_dfn_well_correction/output/candidate_cheye1_multiscale_rebalance_v1/00_step7d_initial_input_all_raw_time.vtk`
- `step8_dfn_well_correction/output/candidate_cheye1_multiscale_rebalance_v1/05_step8_added_well_control_only_raw_time.vtk`
- `step8_dfn_well_correction/output/candidate_cheye1_multiscale_rebalance_v1/well_corrected_dfn_fracture_patches.csv`
- `step8_dfn_well_correction/output/candidate_cheye1_multiscale_rebalance_v1/well_corrected_dfn_raw_time.vtk`
- `step8_dfn_well_correction/output/candidate_cheye1_multiscale_rebalance_v1/step8_well_correction_summary.json`

纠偏规则：

- 原始断层主面不参与井控移动。
- 成像测井真实裂缝为硬约束，但新增片大小不能超过中尺度上限。
- 常规测井预测裂缝只补充小尺度背景裂缝。
- 井控新增片中心允许在井旁小范围随机偏移，不全部压在井轨迹上。
- 井控 patch 尺寸由井点裂缝密度控制，但必须设置上限，避免超长片。

通过标准：

- `00_step7d_initial_input_all_raw_time.vtk` 与 Step7D 原始 VTK 几何一致。
- `05_step8_added_well_control_only_raw_time.vtk` 中新增片数量和位置符合井控逻辑，不全部机械贴井。
- 不出现异常超长片。
- 大尺度原始断层 hard prior 不被移动或删除。

不通过处理：

- 如果 00 输入 VTK 与 Step7D 不一致，先修 CSV/VTK 读写，不继续井控。
- 如果 05 新增片全部贴井，恢复井旁偏移策略。
- 如果出现超长片，检查单位换算和 density-size 映射。

## 14. 阶段 10：Step9 剖面图和最终 QC

目标：输出最终 8 张核心剖面图，并能说明 DFN 与地震解释成果和井上真实解释的一致性。

主要输入：

- Step8 修正后 DFN。
- 相干体、蚂蚁体。
- 原始成像测井裂缝解释。
- 原始断层 `FaultStick-GeoEast.dat` 或 Step6C 中筛选的原始断层 surface。

建议脚本：

- 修改现有：`step9_section_visualize`
- 新配置：`formal_candidate_cheye1_step9_multiscale_rebalance_v1.json`

输出：

- `step9_section_visualize/output/candidate_cheye1_multiscale_rebalance_v1/cheye1_dfn_coherence_sections/`
- `step9_section_visualize/output/candidate_cheye1_multiscale_rebalance_v1/cheye1_dfn_coherence_sections/section_qc_summary.json`

必须保留 8 张图：

- XZ DFN 剖面。
- YZ DFN 剖面。
- XZ 相干体剖面。
- YZ 相干体剖面。
- XZ 相干体 + DFN 剖面。
- YZ 相干体 + DFN 剖面。
- XZ 井周 200 m 相干体 + DFN 剖面。
- YZ 井周 200 m 相干体 + DFN 剖面。

展示规则：

- DFN 画裂缝片与剖面的交线，不只画中心点。
- 原始成像测井裂缝画真实解释片投影，用于与生成 DFN 对比。
- 原始断层叠加必须来自原始解释轨迹/曲面，不使用 Step7C 后处理断层片冒充原始解释。
- 相干体图件中黑灰色不连续带用于对比断裂发育区；水平黑条带要在解释中说明可能是层位界面，不作为主要裂缝证据。
- 图例按尺度区分 small/medium/large，并保留成像测井真实裂缝和原始断层标识。

通过标准：

- 输出 8 张图齐全。
- `section_qc_summary.json` 统计每个剖面上的 small/medium/large 交线数量、原始断层可见段数、成像测井真实裂缝可见数量。
- 如果当前剖面不经过原始断层，summary 必须明确记录，不能强行画后处理断层。
- 中尺度在剖面上应比当前版本更贴近蚂蚁体高值和局部相干体黑灰断续带。

不通过处理：

- 如果剖面上看不到大中尺度，先检查剖面 half-width 和 patch-plane intersection。
- 如果断层 overlay 为空，检查当前剖面是否确实不切原始断层；不要用生成的 DFN 断层片替代原始断层。
- 如果小尺度遮挡主体，Step9 降低小尺度透明度或抽样显示，不修改原始 DFN 数据。

## 15. 执行顺序

严格按以下顺序执行，每一步先看 QC，再进入下一步：

1. 阶段 0：输入和坐标 QC。
2. 阶段 1：Step6A 小尺度背景密度。
3. 阶段 2：Step6B 中尺度裂缝带先验。
4. 阶段 3：Step6C 大尺度断层/断裂带先验。
5. 阶段 4：Step6D 多尺度证据包。
6. 阶段 5：Step7A 小尺度 DFN。
7. 阶段 6：Step7B 中尺度 DFN。
8. 阶段 7：Step7C 大尺度 DFN。
9. 阶段 8：Step7D 多尺度融合。
10. 阶段 9：Step8 测井纠偏。
11. 阶段 10：Step9 剖面图和最终 QC。

阶段性提交建议：

- 完成 Step6A/B/C/D 且 QC 通过后提交一次。
- 完成 Step7A/B/C/D 且 VTK 可视化通过后提交一次。
- 完成 Step8/Step9 且最终 8 图生成后提交一次。

## 16. 验收指标

### 16.1 数据一致性指标

- 所有 SGY/NPZ/VTK 的 XY/T 坐标一致。
- 所有 VTK 可被 ParaView 正常打开，无 `nan` cell attribute。
- Step8 的 `00_step7d_initial_input_all_raw_time.vtk` 与 Step7D 输出几何一致。

### 16.2 尺度分离指标

- 小尺度数量最多、面积最小。
- 中尺度数量适中、沿 corridor 成带。
- 大尺度数量最少、面积最大、与原始断层或陡倾低相干候选对应。
- `PatchAreaM2` 在三个尺度上有明显分布差异。

### 16.3 地震解释一致性指标

- 中尺度 patch 与蚂蚁体高值区重合率高于当前版本。
- 大尺度 patch 与原始断层/相干体竖向低值不连续带有空间对应。
- 水平层界面低相干不应大量转化为大尺度断层。

### 16.4 井控一致性指标

- 成像测井真实解释片与井周生成 DFN 能在剖面上对比。
- 常规测井预测只补充小尺度背景，不改动大断层。
- 井控新增片不全部机械贴井轨迹，也不出现异常超长片。

## 17. 风险和修正策略

- 风险：Step6B 中尺度仍产生一个覆盖全区的大连通体。
  处理：先做 component 切分和薄层过滤，不直接进入 Step7B。

- 风险：Step6C 把水平地层界面识别成断层。
  处理：增加垂向连续性、倾角和时间厚度过滤；必要时先只保留原始断层 + 高置信 inferred fault。

- 风险：单元断层 patch 与 `FaultStick-GeoEast.dat` 坐标或时间不一致。
  处理：不复用旧 patch，改为从 GeoEast 原始数据重建断层面。

- 风险：Step7B 视觉上仍像散乱裂缝片。
  处理：降低片间距、增强同一 component 内方向平滑、适度增加重叠率，而不是单纯增大片尺寸。

- 风险：Step7C 大尺度仍不明显。
  处理：优先保留原始 panel 多边形和面积，不把断层面简化成小矩形片。

- 风险：Step8 破坏 Step7D 几何。
  处理：保留 00/05/最终三个 VTK，中间检查不通过则停止，不进入 Step9。

## 18. 方案自检结论

本方案对当前主要漏洞做了约束：

- 已避免覆盖旧结果：所有新输出使用 `candidate_cheye1_multiscale_rebalance_v1`。
- 已避免把所有证据混成单一密度体：Step6A/B/C/D 保留小、中、大尺度通道和证据来源。
- 已避免把水平低相干层界面直接当断层：Step6B/6C 都加入薄层、倾角和垂向连续性过滤。
- 已考虑原始断层解释是硬约束：Step6C/Step7C 优先使用原始 fault patch/panel 和 `FaultStick-GeoEast.dat`。
- 已考虑原始断层不完整：Step6C 额外识别高置信未解释断层候选。
- 已修正小尺度过稀疏问题：Step7A 放宽候选分位数，并以数量和面积分布作为 QC。
- 已修正中尺度过大且不连续问题：Step7B 增加 corridor component 切分、局部方向平滑、片间距和面积上限。
- 已修正大尺度面积偏小问题：Step7C 优先保留原始 panel 多边形，不再只生成少量小矩形。
- 已保留故障定位能力：Step8/Step9 都要求中间 VTK 和 summary，便于定位问题发生在哪一步。

仍需注意的执行边界：

- 如果阶段 0 发现输入坐标不一致，必须先修坐标，不应继续调参数。
- 如果 Step6B/6C 的 QC 不通过，不应直接进入 Step7 生成 DFN。
- 如果最后剖面效果不好，应先判断问题来自 Step6 证据体、Step7 几何化还是 Step9 投影，不应一次性改多个环节。
