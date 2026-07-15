# Step6 多尺度裂缝预测改造实现计划

> 下一轮可落地执行方案见：
> `STEP6_STEP7_MULTISCALE_REBALANCE_EXECUTION_PLAN.md`。
> 该文档细化了 `candidate_cheye1_multiscale_rebalance_v1` 的输入 QC、Step6A/B/C/D、Step7A/B/C/D、Step8、Step9 执行顺序、输出目录、验收标准和不通过处理策略。

## 1. 改造目标

当前旧 Step6/旧 Step6B 的问题不是“能不能生成密度体”，而是生成出来的裂缝密度和地震解释成果之间的关系不够清楚。后续改造目标是把裂缝预测拆成大、中、小三个尺度通道：

- 大尺度：断层和大断裂带，重点和原始断层解释、相干体低值不连续带对应。
- 中尺度：裂缝带和小断层候选，重点和蚂蚁体高值、局部低相干带、曲率异常对应。
- 小尺度：背景裂缝和测井尺度裂缝，重点和常规测井预测、成像测井真实解释、曲率异常对应。

最终 Step7/Step8/Step9 不再只面对一个“总裂缝密度”，而是能够知道每个裂缝片来自哪个尺度、由什么证据支持、应该以什么方式展示和评价。

## 2. 地质解释口径

本项目后续汇报建议采用以下口径：

- 相干体主要反映地震同相轴不连续，适合识别大尺度断层、断裂带和强不连续边界。
- 蚂蚁体是在不连续属性基础上做追踪和增强，适合突出断层、小断层、裂缝带等线性或带状不连续结构。
- 曲率体反映地层弯曲和局部应变异常，适合作为小断层和裂缝易发区的辅助证据，但不能直接等同于小裂缝。
- 原始断层解释是地质人员直接解释成果，属于硬约束；未解释的小断层可以由相干体、蚂蚁体、曲率体联合识别。
- 测井裂缝解释和常规测井预测主要对应井周小尺度裂缝和背景裂缝密度。

需要避免的说法：

- 不应说“相干体就是大尺度裂缝、蚂蚁体就是中尺度裂缝、曲率体就是小尺度裂缝”。
- 更准确的说法是“不同地震属性对不同尺度裂缝/断裂具有不同敏感性，需要联合约束”。

## 3. 当前主要问题

### 3.1 Step6 二维密度不适合作为主控 DFN 输入

Step6 当前输出的是每个地震道、每个层段一个密度值。它可以表达“这个位置这个层段整体裂缝多不多”，但不能表达层内哪一段时间深度裂缝最发育，也不能表达断裂带的空间连通性。

后续定位：

- Step6 保留为小尺度背景裂缝趋势通道。
- 不再让 Step6 的二维层段密度直接主控大中尺度 DFN。

### 3.2 Step6 近井预测与真实井吻合较弱

已有 summary 中近井校验表现较差：

- 沙三段预测密度与最近真实井密度相关性约为负相关。
- 沙四段预测密度与最近真实井密度相关性很弱。

说明当前二维密度模型不能很好解释井点约束，后续必须增加按井验证和近井一致性 QC。

### 3.3 Step6B 三维密度被虚拟井样本强烈主导

Step6B 训练样本中虚拟井数量远大于真实井：

- 真实井样本约 37.6 万行。
- 虚拟井样本约 940 万行。

虽然真实井单点权重更高，但虚拟井数量太大，总体上仍可能主导模型。后续需要按源井、按 SourceKind 做总权重归一，避免“虚拟井扩出来的规律”压过真实井。

### 3.4 当前验证方式偏乐观

Step6B 当前 validation 是随机行抽样。由于相邻时间点、相邻虚拟井、同源井样本高度相似，随机验证不能代表模型对未知井或未知区域的泛化能力。

后续需要增加：

- 按真实井留一验证。
- 按 SourceWellName 分组验证。
- 近井预测与真实井标签相关性检查。

### 3.5 地震先验后处理过粗

当前旧 Step6B 后处理是先生成三维密度体，再用相干体、蚂蚁体、曲率体构造 SeismicPrior 对密度进行增强。

问题：

- 后处理属于“预测后修正”，不是模型源头约束。
- 低相干中可能包含横向层界面、噪声和大范围背景异常。
- 当前低相干连通体可能出现覆盖整个 demo 区的大块异常，容易把非断裂区域整体增强。

后续需要将地震先验拆成尺度通道，不再直接把所有先验混成一个密度增强因子。

## 4. 新的 Step6 总体结构

建议将 Step6 重构为以下四个子步骤。

### 4.1 Step6A 小尺度背景裂缝密度

用途：

- 对应测井尺度裂缝和背景裂缝。
- 为 Step7 生成小尺度离散裂缝片提供密度基础。

主要输入：

- Step4 常规测井裂缝预测结果。
- Step3 成像测井真实裂缝解释。
- Step5 真实井和虚拟井统一样本。
- 曲率体、层位相对位置、常规地震属性。

主要输出：

- `small_background_density.sgy`
- `small_background_density_summary.json`
- `small_background_density_qc.json`

关键逻辑：

- 真实井样本权重大于虚拟井。
- 虚拟井按源井归一，不能按行数直接主导训练。
- 曲率体只作为小尺度裂缝易发区辅助特征。
- 输出密度代表背景裂缝发育程度，不负责大断层和中尺度断裂带。

### 4.2 Step6B 中尺度裂缝带先验

用途：

- 对应中尺度裂缝带、小断层组合、断裂损伤带。
- 为新版 Step7 生成成带、成组、方向相对稳定的中尺度裂缝片提供先验。

主要输入：

- 蚂蚁体高值。
- 局部低相干异常。
- 曲率高值异常。
- 距离原始断层的距离。

主要输出：

- `medium_corridor_prior.sgy`
- `medium_corridor_skeleton.vtk`
- `medium_corridor_orientation.npz`
- `medium_corridor_summary.json`

关键逻辑：

- 蚂蚁体高值是主要证据。
- 低相干和曲率作为辅助证据。
- 单独的蚂蚁体噪声不直接增强，至少需要和低相干、曲率、断层邻近性中的一项共同支持。
- 对连通带提取骨架或脊线，给新版 Step7 提供连续性和方向。

### 4.3 Step6C 大尺度断层/断裂带先验

用途：

- 对应大尺度断层和大断裂带。
- 为新版 Step7 生成大尺度连续断裂片或断裂带提供硬约束/强约束。

主要输入：

- 原始断层解释 `FaultStick-GeoEast.dat`。
- 相干体低值异常。
- 蚂蚁体高值作为辅助。
- 层位 T4-T7。

主要输出：

- `large_fault_prior.sgy`
- `large_fault_distance.sgy`
- `large_fault_orientation.npz`
- `large_fault_candidates.vtk`
- `large_fault_prior_summary.json`

关键逻辑：

- 原始断层解释为 hard prior。
- 原始断层附近密度/先验增强，但不把断层本身当作普通小裂缝。
- 对没有原始解释的小断层，使用低相干垂向不连续带识别候选。
- 过滤横向层界面：横向连续、垂向厚度很薄的低相干带不应进入大断层先验。

### 4.4 Step6D 最终小尺度衍生密度修正

用途：

- 在不改变大/中尺度断层和裂缝带本体的前提下，利用 Step6B/Step6C 的构造位置修正小尺度背景裂缝密度。
- 为 Step7A 输出最终小尺度采样密度，同时为 Step7B/Step7C 保留中/大尺度先验索引。

主要输入：

- `small_background_density.sgy`
- `large_fault_prior.sgy`
- `medium_corridor_prior.sgy`
- 大/中尺度方向和连通结构文件。

主要输出：

- `multiscale_density_bundle.json`
- `final_small_density.sgy`
- `final_small_candidate_mask.sgy`
- `scale_label.sgy`
- `scale_confidence.sgy`
- `multiscale_density_qc.json`

关键逻辑：

- Step6A 输出原始小尺度背景密度，不直接考虑大/中尺度影响。
- Step6D 读取 Step6B/Step6C，对中尺度裂缝带和大尺度断层周边的损伤带做小尺度密度增强。
- Step6D 输出的 `final_small_density` 仍然只表示小尺度/衍生小裂缝密度，不包含大/中尺度裂缝本体。
- 大尺度断层本体由 Step7C 表达，中尺度裂缝带本体由 Step7B 表达，Step7A 只读取 Step6D 的最终小尺度密度。
- `scale_label` 和 `scale_confidence` 只用于记录大/中/小尺度空间关系和 QC，不允许把三类证据压成一个来源不明的单通道最终密度。

## 5. 推荐实施顺序

### 阶段 0：只做 QC，不改结果

目标：

- 先明确当前旧 Step6/旧 Step6B 到底哪里不匹配。

新增脚本建议：

- `step6b_demo_density_volume_3d/qc_multiscale_density_vs_attributes.py`

输出：

- `step6_multiscale_qc_current.json`
- `density_vs_coherence_anttrack_curvature.csv`
- `top_density_attribute_overlap.json`

检查内容：

- 井点 Density 与 Coherence、AntTrack、Curvature 的相关性。
- Step6B 高密度 top 1%、top 5% 与低相干、蚂蚁体高值、曲率高值的重合率。
- 高密度体中横向连片和垂向连片比例。
- 不同 SourceKind 的训练权重占比。
- 按井留一验证的基线结果。

完成标准：

- 明确当前小尺度、中尺度、大尺度分别差在哪里。
- 不改变任何已有输出。

### 阶段 1：重构小尺度 Step6A

目标：

- 先把背景裂缝密度做可靠，不急着处理大中尺度。

改动点：

- 从当前旧 Step6B 训练逻辑中拆出小尺度模型。
- 加入按源井归一的样本权重。
- 加入按真实井留一验证。
- 输出 `small_background_density.sgy`。

完成标准：

- 真实井附近预测密度与真实井密度相关性不低于当前 Step6。
- 按井留一验证结果进入 summary。
- 输出仍为 SGY，不输出大体量 CSV。

### 阶段 2：构造中尺度 Step6B

目标：

- 用蚂蚁体为主识别裂缝带和小断层候选。

改动点：

- 蚂蚁体高值归一化。
- 低相干局部异常和曲率异常联合约束。
- 提取连通带、骨架、局部方向。

完成标准：

- 中尺度先验和蚂蚁体高值带空间上有明显对应。
- 不把孤立噪声点当成裂缝带。
- 输出 `medium_corridor_skeleton.vtk` 可视化可检查。

### 阶段 3：构造大尺度 Step6C

目标：

- 用原始断层解释和相干体低值识别大尺度断裂先验。

改动点：

- 新增原始 FaultStick 读取和投影。
- 原始断层解释生成 hard prior。
- 低相干识别未解释断层候选。
- 增加横向层界面过滤。

完成标准：

- 原始断层附近 prior 明显增强。
- 横向层界面不被大量识别成断层。
- 输出 `large_fault_candidates.vtk` 可在 ParaView 检查。

### 阶段 4：最终小尺度衍生密度修正 Step6D

目标：

- 将 Step6B 中尺度裂缝带和 Step6C 大尺度断层的损伤带影响加入小尺度背景密度。
- 输出 Step7A 使用的最终小尺度密度，同时保留 Step7B/Step7C 使用的中/大尺度先验索引。

改动点：

- 输出多通道 SGY/NPZ/JSON。
- 生成 `final_small_density` 和 `final_small_candidate_mask`，作为 Step7A 输入。
- 生成 `scale_label`，记录 large / medium / small 的空间重叠和优先级。
- 对中/大尺度核心区做防重复控制：不把大断层面或中尺度裂缝带本体直接变成小尺度高密度实心块，只增强周边损伤带。

完成标准：

- Step7A 可以只读取 `final_small_density` 生成小尺度裂缝。
- Step7B/Step7C 继续分别读取 Step6B/Step6C 结果，不从 Step6D 的小尺度密度反推中/大尺度裂缝。
- QC 能统计每个尺度占比、密度分布、地震属性重合率。

### 阶段 5：改造新版 Step7

目标：

- DFN 生成方式按尺度分开。

改动点：

- 大尺度：生成少量、长、连续、方向稳定的断裂片或断裂带。
- 中尺度：沿 corridor skeleton 生成成带裂缝片。
- 小尺度：基于 small density 生成离散背景裂缝片。
- 每个裂缝片写入 `FractureScale`、`EvidenceSource`、`ScaleConfidence`。

完成标准：

- ParaView 中能一眼区分大中小尺度。
- 大中尺度具有连续性，小尺度作为背景散布。
- 新流程不再以旧 `step7` 二维密度细化和旧 `step7c` 后融合为必经步骤；原目录只作为历史回退和对比。

### 阶段 6：改造 Step8

目标：

- 测井纠偏只主要影响小尺度和部分中尺度，不直接移动大尺度断层硬约束。

改动点：

- `large_fault` 不参与井控移动。
- 成像测井真实裂缝可修正小尺度和井旁中尺度裂缝。
- 常规测井预测裂缝只补充小尺度背景裂缝。

完成标准：

- 原始断层硬约束不被井控破坏。
- 井控新增片不再过于规则贴井轨迹。

### 阶段 7：改造 Step9

目标：

- 最终图能说明 DFN 与相干体、蚂蚁体、成像测井、原始断层解释的一致性。

改动点：

- 大尺度裂缝与原始断层/相干体对比。
- 中尺度裂缝与蚂蚁体/低相干带对比。
- 小尺度裂缝与成像测井/常规测井预测对比。
- 图例按尺度区分，不再所有裂缝片一种颜色。

完成标准：

- 8 张 demo 剖面保留。
- 增加多尺度 QC summary。
- 若当前剖面没有原始断层经过，summary 明确写出“无原始断层轨迹经过”，不能误画后处理断层片冒充原始解释。

## 6. 建议输出目录命名

后续不要覆盖已有结果，建议新版本统一使用：

- Step6A：`output/candidate_cheye1_multiscale_v1/small_background/`
- Step6B：`output/candidate_cheye1_multiscale_v1/medium_corridor/`
- Step6C：`output/candidate_cheye1_multiscale_v1/large_fault/`
- Step6D：`output/candidate_cheye1_multiscale_v1/fused_multiscale/`
- Step7：`output/candidate_cheye1_multiscale_v1/`
- Step8：`output/candidate_cheye1_multiscale_v1/`
- Step9：`output/candidate_cheye1_multiscale_v1/cheye1_dfn_coherence_sections/`

## 7. 必须保留的回退点

当前 GitHub 节点：

- 分支：`second-round-optimization`
- 提交：`a8576f7 Update well correction and fault-aware section QC`

后续每完成一个阶段，建议单独提交一次，避免大改失败后无法回退。

## 8. 第一轮具体执行清单

建议下一步只做阶段 0，不改模型：

1. 新增 `qc_multiscale_density_vs_attributes.py`。
2. 读取当前旧 Step6B 原始密度、seismic prior 版本密度、相干体、蚂蚁体、曲率体。
3. 统计高密度区与三类属性的重合率。
4. 统计井点 Density 与属性体的相关性。
5. 统计横向低相干带和垂向低相干带比例。
6. 输出 `step6_multiscale_qc_current.json`。
7. 根据 QC 决定 Step6A/Step6B/Step6C 的具体阈值和权重。

阶段 0 完成后，再开始改 Step6A。不要直接改新版 Step7，否则后续很难判断问题来自密度体还是 DFN 生成逻辑。

## 9. 复查结论和执行边界

本轮改造不建议继续在现有单一密度体上反复调权重。当前问题的核心是尺度混在一起：井上预测主要约束小尺度背景裂缝，相干体和蚂蚁体主要约束断裂/裂缝带，原始断层解释又是硬约束。如果继续把这些信息压成一个密度体，再让 Step7 统一采样，就很难同时满足“井点一致”“地震解释一致”“DFN 视觉上有大中小尺度差异”三个目标。

后续必须按以下边界执行：

- 先做阶段 0 QC，确认现有密度体和相干体、蚂蚁体、曲率体、井点标签的关系，不直接改 DFN。
- Step6A 只解决小尺度背景裂缝，不承担断层和中尺度裂缝带识别。
- Step6B 以蚂蚁体高值为主，联合低相干和曲率，重点处理中尺度裂缝带。
- Step6C 以原始断层和低相干陡向不连续为主，重点处理大尺度断裂。
- Step6D 保留尺度标签和证据来源，不能只输出一个没有来源信息的最终密度体。
- 新版 Step7 按尺度生成 DFN，大尺度少而连续，中尺度成带，小尺度离散补充。
- Step8 井控纠偏不能破坏原始断层硬约束，主要修正井旁小尺度和局部中尺度裂缝。
- Step9 的图件必须分别说明 DFN 与相干体、蚂蚁体、成像测井、原始断层解释的对应关系，不能用后处理生成的断层片冒充原始断层解释。

如果阶段 0 QC 证明当前旧流程三维高密度体与蚂蚁体/低相干几乎无关，则优先重构 Step6B 和 Step6C；如果井点处预测密度与真实裂缝密度也明显不相关，则 Step6A 也需要同步重训或重加权。这样可以避免在错误密度体基础上继续优化 Step7/Step8。
