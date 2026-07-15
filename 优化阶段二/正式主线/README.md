# 优化阶段二 正式主线

本目录存放第二轮优化当前确认保留的正式代码。目标是形成一条可复现、可汇报、可回退的正式流程：从井数据与地震属性样本构造，到二维/三维裂缝密度建模，再到 DFN 细化、断层解释约束融合、井控纠偏和剖面展示。

## 当前主线口径

当前实现不是单一路径，而是保留两类密度与 DFN 结果，并在 Step7C 融合：

- `Step6/Step7`：二维层段裂缝密度表格与二维密度 DFN 细化结果。二维结果提供平面分布补充，但倾向倾角主要来自默认模板，不能作为最终方向硬依据。
- `Step6B/Step7B`：三维裂缝密度体与三维密度 DFN 细化结果。当前推荐使用低相干/地震解释先验修正后的三维密度体，以及 `detail_preserve_v4` 版本 DFN。
- `Step7C`：先融合 Step7 二维 DFN 与 Step7B 三维 DFN，再把地质人员解释的断层 patch/panel 作为硬约束纳入 DFN。
- `Step8`：基于真实井、预测井控点和成像测井真实解释裂缝进行井控纠偏。
- `Step9`：生成车页1导眼井 demo 剖面图、相干体/DFN 叠合图和对比图。

当前蚂蚁体解释口径：蚂蚁体高值表示裂缝/断裂发育更强，低值表示不发育。

当前相干体解释口径：低相干异常更接近断裂/不连续带；但横向连续低相干条带可能是地层界面，不应全部解释为有效裂缝。

## 新版多尺度 Step6 口径

后续多尺度流程中，Step6 不再输出一个“万能裂缝密度体”，而是按尺度拆分证据：

- `Step6A`：原始小尺度背景密度。只表达由测井和虚拟测井推演得到的背景小裂缝发育趋势，不直接考虑大/中尺度断层和裂缝带影响。
- `Step6B`：中尺度裂缝带先验。以蚂蚁体高值为主，低相干和曲率作为辅助，输出中尺度裂缝带的位置、mask、component 和局部方向统计。
- `Step6C`：大尺度断层/断裂带先验。原始断层解释和单元断层 patch 为硬约束，低相干陡向不连续用于补充未解释断层候选，并过滤水平层界面。
- `Step6D`：最终小尺度衍生密度修正。读取 Step6A/B/C，把中尺度裂缝带和大尺度断层周边损伤带对小裂缝的增强加入 Step6A 背景密度，输出 `final_small_density` 给 Step7A 使用。

关键约束：Step6D 输出的最终密度仍然只表示小尺度/衍生小裂缝密度，不包含大/中尺度裂缝本体。大尺度本体由 Step7C 表达，中尺度本体由 Step7B 表达，小尺度背景和衍生裂缝由 Step7A 表达。

## 推荐执行链路

```text
step1_surface_framework
  -> step2_real_well_t4_t7_samples
  -> step3_imaging_supervision_samples
  -> step4_expert_real_well_prediction
  -> step5a_single_source_virtual_wells
  -> step5b_unified_samples_t4_t7
  -> step6_demo_density_volume
  -> step6b_demo_density_volume_3d
  -> step7_initial_dfn
  -> step7b_initial_dfn_3d
  -> step7c_fused_dfn
  -> step8_dfn_well_correction
  -> step9_section_visualize
```

其中当前车页1导眼 demo 的推荐核心链路是：

```text
Step7 2D DFN:
  step7_initial_dfn/output/candidate_cheye1/initial_dfn_fracture_patches.csv

Step7B 3D DFN:
  step7b_initial_dfn_3d/output/candidate_cheye1_seismic_prior_lowcoh_v1_detail_preserve_v4/initial_dfn_fracture_patches.csv

Step7C fused with fault postfusion:
  step7c_fused_dfn/configs/formal_fused_candidate_cheye1_detail_preserve_v4_fault_postfusion_v2.json
```

## 步骤说明

### Step1 层位框架

目录：`step1_surface_framework`

作用：

- 读取 T4-T7 层位相关数据。
- 给井样本补充层位归属与 T4-T7 时间窗。
- 为后续 T4-T7 样本构造提供统一层位框架。

主要脚本：

- `run_surface_classifier.py`
- `attach_sample_horizons.py`
- `surface_tools.py`

### Step2 真实井 T4-T7 样本

目录：`step2_real_well_t4_t7_samples`

作用：

- 整理真实井在 T4-T7 层段内的样本。
- 作为常规测井预测、虚拟井构造和密度体建模的数据基础。

推荐配置：

- `configs/formal_all_wells.json`

### Step3 成像测井监督样本

目录：`step3_imaging_supervision_samples`

作用：

- 重构成像测井真实裂缝解释样本。
- 当前重点保留车页1导眼井真实成像测井段裂缝标签。
- 后续 Step8/Step9 用它与预测 DFN 进行对比。

主要脚本：

- `build_formal_rebuild_groups.py`
- `validate_formal_rebuild_delivery.py`

### Step4 常规测井裂缝预测

目录：`step4_expert_real_well_prediction`

作用：

- 基于专家库/常规测井曲线进行真实井轨迹上的裂缝密度与裂缝点预测。
- 当前第二轮优化未重新补充新的测井曲线数据，常规测井预测仍沿用已确认的正式输出。

推荐配置：

- `configs/formal_expert_prediction.json`

### Step5A 单源虚拟井

目录：`step5a_single_source_virtual_wells`

作用：

- 围绕真实井构造周边虚拟井。
- 虚拟井不是简单复制真实井，而是综合地震属性相似性、空间距离和可信度进行调整。
- 距离真实井越远可信度越低；地震属性越相似可信度越高。

推荐配置：

- `configs/formal_single_source_virtual_wells.json`

### Step5B 统一 T4-T7 密度样本

目录：`step5b_unified_samples_t4_t7`

作用：

- 合并真实井与虚拟井样本。
- 输出二维/三维裂缝密度建模使用的统一训练样本。

推荐配置：

- `configs/formal_unified_t4_t7_density_samples.json`

### Step6 二维 demo 裂缝密度

目录：`step6_demo_density_volume`

作用：

- 构造 demo 区域二维层段裂缝密度结果。
- 本质是 T4-T7 层段内的二维密度表格，每个位置表示该点整体裂缝密度。

推荐配置：

- `configs/formal_candidate_cheye1_density_volume.json`

### Step6B 三维裂缝密度体

目录：`step6b_demo_density_volume_3d`

作用：

- 基于统一样本和地震道网生成三维裂缝密度体。
- 输出采用 SGY 格式，继承原始地震 trace header，避免 CSV 体积过大。
- 后处理阶段引入地震解释先验，当前重点是低相干约束，同时注意压制地层界面造成的横向低相干条带误导。

主要脚本：

- `build_candidate_3d_density_sgy.py`
- `postprocess_density_sgy_lowcoh.py`
- `postprocess_density_sgy_seismic_prior.py`

当前推荐配置：

- `configs/formal_candidate_cheye1_3d_density_seismic_prior_lowcoh_v1_postprocess.json`

### Step7 二维密度 DFN 细化

目录：`step7_initial_dfn`

作用：

- 从二维密度结果生成初始 DFN。
- 当前二维 DFN 主要提供平面裂缝分布补充。
- 检查结果显示，二维 DFN 并非水平片，但倾向倾角来源为默认层模板，因此后续需要由 Step7C 使用三维 DFN 邻域方向进行纠正。

推荐配置：

- `configs/formal_initial_dfn_candidate_cheye1.json`

### Step7B 三维密度 DFN 细化

目录：`step7b_initial_dfn_3d`

作用：

- 从三维密度体生成三维 DFN。
- 当前推荐版本强调细节保留、多尺度裂缝片、面积属性和用于 ParaView 渲染的几何属性。
- 结果中包含 `PatchAreaM2`、`PatchEquivalentRadiusM` 等字段，便于按实际大小着色或筛选。

当前推荐配置：

- `configs/formal_initial_dfn_3d_candidate_cheye1_seismic_prior_lowcoh_v1_detail_preserve_v4.json`

当前推荐输出：

- `output/candidate_cheye1_seismic_prior_lowcoh_v1_detail_preserve_v4/initial_dfn_fracture_patches.csv`
- `output/candidate_cheye1_seismic_prior_lowcoh_v1_detail_preserve_v4/initial_dfn_raw_time.vtk`

### Step7C 二维/三维/断层融合 DFN

目录：`step7c_fused_dfn`

作用：

- 融合 Step7 二维 DFN 与 Step7B 三维 DFN。
- 使用三维 DFN 邻域方向纠正二维 DFN 的倾向倾角。
- 去除与三维 DFN 重复的二维补充片。
- 基于真实断层 patch/panel 做断层后融合。

当前推荐脚本：

- `build_fused_dfn_with_fault_postfusion.py`

当前推荐配置：

- `configs/formal_fused_candidate_cheye1_detail_preserve_v4_fault_postfusion_v2.json`

当前推荐输出目录：

- `output/candidate_cheye1_detail_preserve_v4_fault_postfusion_v2`

该目录保留三个核心 VTK 对比文件：

- `fracture_only_step7_step7b_fused_raw_time.vtk`：仅 Step7 二维 + Step7B 三维裂缝融合结果。
- `fault_only_and_influence_raw_time.vtk`：仅断层 surface、断层诱导裂缝和断层影响带内 shrink 片。
- `fused_initial_dfn_raw_time.vtk`：最终融合结果。

附加说明：

- `SourceDensity` 保留原始密度值。
- `fault_surface` 作为硬约束，`SourceDensity` 会被赋高值 `999`，不建议直接用该字段给最终融合结果着色。
- ParaView 推荐使用 `SourceDensityRender` 或 `SourceDensityRenderNorm` 进行密度着色，避免硬约束值拉爆色标。
- `PatchAreaM2` 用于裂缝片面积渲染，当前已在仅裂缝融合 VTK 中补齐。
- `fault_surface_only_raw_time.vtk` 可用于单独查看断层 surface 硬约束片。
- `raw_fault_surface_selected_patches_unprocessed.vtk` 可用于查看未平面化、未拆分的原始断层面。

断层融合逻辑：

- `regional_fault_panels` 用于计算断层影响带、核心带删除、过渡带 shrink/方向混合和诱导裂缝生成。
- `fault_surface_fragments` 用于显示断层 surface 硬约束。当前 surface fragment 倾向倾角来自原始断层 patch 局部拟合平面。
- 原始断层 patch 的未处理合并面已可作为 QC 参照，避免过度简化导致主连通性误判。

### Step8 井控纠偏

目录：`step8_dfn_well_correction`

作用：

- 基于真实井和预测井控点对 DFN 进行纠偏。
- 成像测井真实解释段使用 Step3 的真实裂缝标签作为约束。
- 常规测井段使用 Step4 预测裂缝点作为约束。
- 当前逻辑允许井控裂缝中心在一定范围内偏移，避免所有裂缝片中心机械落在井轨迹上。

主要脚本：

- `correct_dfn_with_well_controls.py`

注意：

- 旧配置仍指向若干历史 Step7C 输出目录。
- 若使用当前最新断层融合结果，需要将 `initial_dfn_csv` 指向 `step7c_fused_dfn/output/candidate_cheye1_detail_preserve_v4_fault_postfusion_v2/fused_initial_dfn_fracture_patches.csv`，并同步 `initial_dfn_summary_json`。

### Step9 剖面可视化

目录：`step9_section_visualize`

作用：

- 生成车页1导眼井过井 XZ/YZ 剖面。
- 生成 DFN、相干体、DFN 叠合相干体、井周小范围对比图。
- 支持叠加成像测井真实裂缝解释片，用于比较预测 DFN 与真实成像解释。

主要脚本：

- `build_cheye1_dfn_coherence_sections.py`
- `build_demo_well_section_visualization.py`
- `build_mine_attribute_section_visualization.py`
- `build_all_area_section_visualization.py`

注意：

- 如果 Step8 输出目录切换到最新 `fault_postfusion_v2` 链路，Step9 配置中的 `input_vtk` 也需要同步指向对应 Step8 输出。

## 当前推荐查看结果

Step7C ParaView 对比：

```text
step7c_fused_dfn/output/candidate_cheye1_detail_preserve_v4_fault_postfusion_v2/fracture_only_step7_step7b_fused_raw_time.vtk
step7c_fused_dfn/output/candidate_cheye1_detail_preserve_v4_fault_postfusion_v2/fault_only_and_influence_raw_time.vtk
step7c_fused_dfn/output/candidate_cheye1_detail_preserve_v4_fault_postfusion_v2/fused_initial_dfn_raw_time.vtk
step7c_fused_dfn/output/candidate_cheye1_detail_preserve_v4_fault_postfusion_v2/fault_surface_only_raw_time.vtk
step7c_fused_dfn/output/candidate_cheye1_detail_preserve_v4_fault_postfusion_v2/raw_fault_surface_selected_patches_unprocessed.vtk
```

ParaView 推荐字段：

- 按裂缝密度着色：优先用 `SourceDensityRender` 或 `SourceDensityRenderNorm`。
- 按裂缝面积着色：用 `PatchAreaM2` 或 `PatchArea`。
- 区分来源：用 `SourceTypeCode`。
- 区分断层关系：用 `FaultRelationCode`。

## 数据依赖

代码本身应尽量只依赖本仓库内脚本，但运行需要外部数据：

- 地震体、相干体、蚂蚁体、曲率体等属性体。
- T4-T7 层位数据。
- 真实井与成像测井解释数据。
- 断层解释数据及断层切割 patch。

当前断层融合使用的切割 patch 路径：

```text
小范围DFN生成/断层裂缝片生成/断层划分切割/fault_patches_out/patches
```

说明：

- 断层切割结果基于 XY 单元划分，不依赖 inline/xline 对齐。
- 已检查 `FaultStick_0604.dat` 与 `FaultStick-GeoEast.dat` 在同一 XY 下时间值一致，因此当前切割 patch 可复用。

## 项目管理原则

本目录只保留：

- 正式脚本。
- 正式配置。
- 步骤说明文档。
- 必要的公共工具代码。

本目录不应保留：

- 临时 smoke 脚本。
- 已废弃验证脚本。
- 与当前正式输入契约不一致的旧原型。
- `__pycache__` 等解释器缓存。
- 大体积生成产物。

`output/` 目录属于生成产物，原则上不上传 GitHub。需要展示或交付时，应单独记录结果目录和生成配置。

## GitHub 上传口径

默认上传：

- `README.md`
- `MIGRATION_RULES.md`
- 各 step 的 `*.py`
- 各 step 的 `configs/*.json`
- 必要的步骤说明文档

默认不上传：

- 各 step 的 `output/`
- `__pycache__/`
- 本地日志
- 历史调试结果
- 大体积 VTK/SGY/CSV 结果

如果某个输出是阶段汇报必须复现的关键节点，应优先上传生成脚本和配置，并在 README 或汇报文档中记录输出路径、生成时间和关键统计值，而不是直接上传大文件。

## 当前待同步事项

- Step8/Step9 的部分配置仍指向历史融合结果目录，需要在确认 `fault_postfusion_v2` 效果后新建对应配置，避免覆盖旧配置。
- 断层 surface 当前仍使用 fragment 显示版本；若后续要求完全保留原始断层面主连通性，应将最终 `fault_surface` 硬约束改为直接使用原始 raw fault mesh。
- 低相干先验仍需继续区分垂向断裂异常与横向地层界面异常，避免把层位界面误增强为裂缝。
