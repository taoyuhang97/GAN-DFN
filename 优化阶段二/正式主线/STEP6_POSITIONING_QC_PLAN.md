# Step6 定位与可靠性确认实施规划

## 1. 本文档目标

本阶段先不继续修改 Step7，也不立即重跑 DFN。目标是把 Step6 的定位说清楚、验证清楚：

- Step6 哪些结果可信，能作为后续 DFN 输入。
- Step6 哪些结果只能作为辅助趋势，不能直接控制裂缝片。
- Step6 与测井标签、相干体、蚂蚁体、曲率体是否一致。
- Step6A/B/C/D 后续各自应该承担什么职责。

最终输出一个明确结论：当前 Step6 原始三维密度体是否只适合作为小尺度背景裂缝，Step6B/6C 是否必须由地震解释属性单独构造。

## 2. Step6 的明确定位

后续不再把 Step6 理解为“一个万能裂缝密度体”。Step6 应定位为多尺度证据准备步骤。

### 2.1 Step6A：小尺度背景裂缝密度

允许表达：

- 测井尺度裂缝的空间趋势。
- 不同层段内部的背景裂缝发育强弱。
- 井间区域的小尺度采样概率。

不允许表达：

- 大尺度断层面。
- 中尺度连续裂缝带的几何形态。
- 裂缝片的最终倾向倾角。

判断标准：

- 与真实井/虚拟井样本中的 `DensityLabel`、`HasFracture` 方向一致。
- 在真实井上不能明显背离井点裂缝密度。
- 高值区可以和低相干、蚂蚁体有一定关系，但不要求完全贴合地震解释带。

### 2.2 Step6B：中尺度裂缝带先验

允许表达：

- 蚂蚁体高值对应的裂缝带、小断层组合、断裂损伤带。
- 局部低相干和曲率异常共同支持的中尺度不连续带。
- 中尺度候选带的位置、宽度、连通性和局部方向。

不允许表达：

- 只靠原始 Step6 高密度强行生成中尺度带。
- 把横向层界面当成中尺度裂缝带。
- 把全区大连通噪声当成一个裂缝系统。

判断标准：

- 中尺度高值必须明显贴近蚂蚁体高值。
- 低相干只作为辅助，不能让横向层界面主导。
- 连通体数量、大小、方向要有差异，不能全区同质化。

### 2.3 Step6C：大尺度断层/断裂带先验

允许表达：

- 甲方/地质人员提供的原始断层解释成果。
- 与相干体低值、蚂蚁体高值共同支持的未解释断层候选。
- 大尺度断层附近的影响带。

不允许表达：

- 把所有低相干黑色条带都当成断层。
- 把原始断层 patch 转成方向错误的小矩形片。
- 把 Step6A 小尺度密度高值直接升级成大断层。

判断标准：

- 原始断层为硬约束，方向和位置必须保留。
- 自动识别断层必须具备垂向连续性，不能只是横向层界面。
- 大尺度候选应在相干体低值上有明显响应。

### 2.4 Step6D：最终小尺度衍生密度修正

允许表达：

- 基于 Step6A 原始小尺度背景密度，叠加 Step6B 中尺度裂缝带和 Step6C 大尺度断层周边损伤带影响。
- 给 Step7A 提供最终小尺度密度 `final_small_density` 和 `final_small_mask`。
- 保留小、中、大三类证据的统一索引、尺度标签和 QC 统计。

不允许表达：

- 把大/中尺度裂缝本体混入小尺度密度。
- 把小、中、大三类证据压成一个来源不明的最终密度。
- 让 Step7B/Step7C 从 Step6D 的小尺度密度反推中/大尺度裂缝。

判断标准：

- 必须保留 `base_small / final_small / medium / large / scale_label / confidence`。
- Step7A 只读 `final_small_density`；Step7B 读 Step6B；Step7C 读 Step6C。

## 3. 需要补齐的 QC 证据链

### 3.1 输入与坐标 QC

目的：排除前置数据错位。

检查内容：

- Step6 密度体、trace mapping、原始地震体的 trace 数和 XY 是否一致。
- 相干体、蚂蚁体、曲率体能否按同一 source trace index 和时间轴插值到 Step6 网格。
- 属性体时间范围是否完整覆盖 Step6 的 `2240-3240 ms`。
- 原始断层和单元断层 patch 是否都在时间域 ms 坐标。

输出建议：

- `step6_positioning_qc/input_alignment_qc.json`
- `step6_positioning_qc/attribute_axis_qc.csv`
- `step6_positioning_qc/fault_alignment_qc.csv`

通过标准：

- XY header 误差中位数为 0 或接近 0。
- 属性体时间范围完全覆盖 Step6。
- 断层 patch 与 `FaultStick-GeoEast.dat` 在同一 XY/T 坐标系。

### 3.2 Step6 来源与训练样本 QC

目的：说明 Step6 密度体到底是怎么学出来的。

检查内容：

- 真实井、虚拟井数量。
- 真实井、虚拟井样本行数和总权重。
- 每个层段样本数量、密度分布、零值比例。
- 虚拟井是否在样本数量上压倒真实井。
- 训练特征是否包含相干体、蚂蚁体、曲率体。

输出建议：

- `step6_positioning_qc/training_sample_qc.json`
- `step6_positioning_qc/source_weight_qc.csv`
- `step6_positioning_qc/layer_density_distribution.csv`

通过标准：

- 能清楚说明模型使用了全部真实井和虚拟井。
- 能量化真实井和虚拟井的权重占比。
- 如果虚拟井总权重过高，需要在结论中标记 Step6A 只能作为背景趋势。

### 3.3 按井验证 QC

目的：检查模型是否真的学到了测井规律，而不是只在随机样本上拟合得好。

检查内容：

- 按真实井留一验证：每次留一口真实井，其他井训练，验证留出井。
- 按 SourceWellName 分组验证：避免同源虚拟井泄漏。
- 真实井附近预测密度与井上 `DensityLabel` 的相关性。
- 真实井裂缝点和非裂缝点的预测密度差异。

输出建议：

- `step6_positioning_qc/leave_one_well_validation.json`
- `step6_positioning_qc/well_holdout_metrics.csv`
- `step6_positioning_qc/real_well_density_match.csv`

通过标准：

- 留一井验证不能明显崩溃。
- 裂缝点预测密度均值应高于非裂缝点。
- 如果只有随机验证好、按井验证差，则 Step6 只能作为小尺度弱趋势，不能强控制 Step7。

### 3.4 井点标签与属性体关系 QC

目的：检查“测井学出的规律”和地震属性是否方向一致。

检查内容：

- `DensityLabel / HasFracture` 与 `Coherence` 的相关性。
- `DensityLabel / HasFracture` 与 `AntTrack` 的相关性。
- `DensityLabel / HasFracture` 与 `CurvatureMax / CurvaturePos` 的相关性。
- 分真实井、虚拟井、沙三段、沙四段分别统计。

输出建议：

- `step6_positioning_qc/well_label_attribute_correlation.json`
- `step6_positioning_qc/well_label_attribute_correlation.csv`

通过标准：

- 与相干体应总体负相关，即低相干更容易对应裂缝。
- 与蚂蚁体应总体正相关，至少真实井上方向不能反。
- 曲率体可以弱相关，但如果完全无关，就只能作为辅助先验。

### 3.5 Step6 密度体与属性体空间重合 QC

目的：检查 Step6 输出是否贴近相干体、蚂蚁体、曲率体。

检查内容：

- Step6 高密度 top 10%、5%、2%、1% 与低相干 top 区域重合率。
- Step6 高密度 top 区域与蚂蚁体高值重合率。
- Step6 高密度 top 区域与曲率体高值重合率。
- 密度与 `LowCoherenceScore / AntTrackScore / CurvatureScore` 的 Pearson 和 Spearman 相关。

输出建议：

- `step6_positioning_qc/top_density_attribute_overlap.json`
- `step6_positioning_qc/density_attribute_correlation.json`
- `step6_positioning_qc/density_attribute_overlap_summary.csv`

通过标准：

- 如果高密度明显富集于低相干，但不富集于蚂蚁体，说明 Step6 更偏相干背景，不适合作为中尺度裂缝带主控。
- 如果高密度与蚂蚁体/曲率体都弱相关，说明中小尺度地震解释约束不足，需要 Step6B/6C 单独构造。

### 3.6 横向层界面误识别 QC

目的：避免把地层界面低相干条带当裂缝。

检查内容：

- 低相干连通体的 XY 范围、时间厚度、垂向连续性。
- 横向跨度大但时间厚度薄的 component 数量。
- 陡倾/垂向连续 component 数量。
- Step6 高密度是否集中在横向薄层低相干上。

输出建议：

- `step6_positioning_qc/lowcoh_component_geometry_qc.json`
- `step6_positioning_qc/lowcoh_components.vtk`

通过标准：

- 横向薄层低相干只能作为层界面，不进入大中尺度裂缝候选。
- 中大尺度候选必须有一定垂向延伸或局部陡倾几何。

## 4. 最终决策规则

完成 QC 后按以下规则给 Step6 定位。

### 4.1 原始 Step6 可作为 Step6A 的条件

满足以下条件即可保留为小尺度背景：

- 坐标和时间轴 QC 通过。
- 真实井上裂缝密度方向没有明显反转。
- 高密度区与低相干或井点裂缝至少有弱到中等一致性。
- 输出密度不全由极少数异常值控制。

如果满足：

- 原始 `candidate_cheye1_3d_predicted_density.sgy` 冻结为 Step6A 输入。
- 后续只对它做归一化、候选采样和小尺度 QC，不再让它主控中大尺度。

### 4.2 需要重训 Step6A 的条件

出现以下任一情况，需要重训或修正 Step6A：

- 按井留一验证明显失败。
- 真实井裂缝点预测密度低于非裂缝点。
- 虚拟井总权重明显压过真实井，且真实井匹配差。
- 分层段后某一层段方向明显反。

重训方向：

- 按源井归一虚拟井权重。
- 增加真实井权重上限和虚拟井总权重上限。
- 用分组验证替代随机验证作为主要评价。

### 4.3 Step6B 必须独立构造的条件

出现以下情况，Step6B 不应读取原始 Step6 高密度作为主控：

- 原始 Step6 高密度与蚂蚁体高值重合弱。
- 中尺度裂缝带在剖面上更接近蚂蚁体/低相干，而不是测井背景密度。
- 原始 Step6 输出过平滑，不能表达连续裂缝带边界。

处理方式：

- Step6B 以蚂蚁体高值为主。
- 低相干和曲率只作为辅助。
- Step6A 只用于判断中尺度带周边是否有小尺度衍生裂缝，不直接决定中尺度带位置。

### 4.4 Step6C 必须独立构造的条件

出现以下情况，Step6C 必须以断层解释和低相干几何为主：

- 原始断层解释已提供。
- 原始 Step6 高密度不能恢复断层面的连续几何。
- 相干体上存在明显陡向低值不连续，但 Step6 密度不连续。

处理方式：

- 原始断层 patch/panel 为硬约束。
- 自动识别小断层候选必须通过垂向连续和非层界面过滤。
- Step6A 不参与大断层面的方向确定。

## 5. 建议实施顺序

### 阶段 1：新增 Step6 定位 QC 脚本

脚本建议：

- `step6b_demo_density_volume_3d/qc_step6_positioning.py`

输出目录：

- `step6b_demo_density_volume_3d/output/candidate_cheye1_step6_positioning_qc_v1/`

执行内容：

- 汇总输入对齐、训练样本、井点标签、属性体重合、低相干 component 几何。
- 不生成新 SGY，不改变已有结果。

### 阶段 2：生成 Step6 定位报告

脚本建议：

- `step6b_demo_density_volume_3d/build_step6_positioning_report.py`

输出：

- `step6_positioning_report.md`
- `step6_positioning_decision.json`

报告必须回答：

- 原始 Step6 是否可信。
- 它可信到哪个尺度。
- Step6A 是否需要重训。
- Step6B 是否必须改为蚂蚁体主控。
- Step6C 是否必须改为断层/低相干几何主控。

### 阶段 3：冻结 Step6A 输入

如果 QC 结论认为原始 Step6 可作为小尺度背景：

- 写入配置：`step6a_input_density = candidate_cheye1_3d_predicted_density.sgy`
- 输出 `step6a_small_background_locked_summary.json`
- README 中明确：Step6A 是小尺度背景，不代表中大尺度断裂带。

如果 QC 不通过：

- 暂停 Step7。
- 先修正训练权重和按井验证，再重训 Step6A。

### 阶段 4：明确 Step6B/6C 输入合同

Step6B 输入合同：

- 必须读取蚂蚁体、相干体、曲率体。
- 必须输出中尺度 component、prior、mask、方向统计。
- 不允许只读取 Step6A 密度直接生成中尺度裂缝。

Step6C 输入合同：

- 必须读取原始断层解释和单元断层 patch。
- 必须输出原始断层 hard prior 和自动识别候选分开的结果。
- 不允许用 Step6A 高密度替代断层解释。

### 阶段 5：再进入 Step7

只有在 Step6 定位报告通过后，才继续 Step7：

- Step7A 只读 Step6D 输出的最终小尺度密度，不直接读 Step6B/Step6C。
- Step7B 主要读 Step6B，生成中尺度裂缝带本体。
- Step7C 主要读 Step6C，不从 Step6A 推断大断层。
- Step7D 只融合和去重，不重新解释尺度。

## 6. 当前已知初步结论

基于当前已有 QC：

- 原始 Step6 不是坏结果，但更像小尺度背景密度和低相干趋势的平滑表达。
- 原始 Step6 与低相干有一定一致性，但与蚂蚁体和曲率体的一致性偏弱。
- 真实井上裂缝密度与相干体负相关、与蚂蚁体弱正相关，方向上基本合理。
- `IsRefinedFracturePoint` 与属性体相关性很弱，说明点级裂缝位置不应直接由当前 Step6 密度体解释。
- 当前 Step6C 自动推断大断层没有保留有效候选，说明不能强行把低相干都当断层。

因此目前更合理的定位是：

- Step6A：保留原始三维密度体作为小尺度背景候选。
- Step6B：另建蚂蚁体主控的中尺度裂缝带先验。
- Step6C：另建原始断层 hard prior + 严格低相干几何筛选。
- Step6D：只修正小尺度衍生密度并保留多尺度索引，不再宣称 integrated density 是唯一最终密度。

## 7. 不应继续做的事情

- 不应继续只调 Step7 裂缝片尺寸来弥补 Step6 证据不清的问题。
- 不应把相干体所有黑色条带都当裂缝。
- 不应把 Step6 原始高密度区直接解释为中大尺度裂缝带。
- 不应把三类尺度融合后只保留一个密度值，丢失来源和尺度标签。
- 不应在没有按井验证的情况下用随机验证 R2 证明模型可靠。
