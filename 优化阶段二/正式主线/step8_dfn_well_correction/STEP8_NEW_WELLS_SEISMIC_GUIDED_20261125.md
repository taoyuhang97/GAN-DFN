# Step8 新增井与地震引导加密

## 版本

- 配置：`configs/formal_mine_multiscale_flow_v1_seg1_step8_new_wells_seismic_guided_v1.json`
- 输出：`output/formal_mine_multiscale_flow_v1_seg1/step8_new_wells_seismic_guided_v1`
- 基础井控：现有 Step8 的 490 个事件/成像控制点
- 新增井：车152、车斜155

## 新增井输入

- 连续预测结果：`formal_new_well_inference_20261125`
- 使用最终细化裂缝点 `all_new_wells_fracture_points.csv`
- 541个细化点按现有 `6 ms` 事件间隔和 `18 ms` 最大跨度规则聚成35个井级事件：
  - 车152：20个事件
  - 车斜155：15个事件
- 35个事件面片强制以事件点为中心，保证面片真正经过对应井轨迹位置。

## 地震引导加密

加密不再使用原径向补偿直接复制面片。新流程为：

1. 在每个井控事件周围建立 0-50、50-100、100-200 m 三个径向带。
2. 每个带生成有限候选位置，中心TWT相对控制点偏移5-8 ms，沿局部倾向方向形成真实三维错位。
3. 在候选位置采样蚂蚁体、相干体和最大曲率，并计算其与井控事件位置地震属性的归一化相似度。
4. 每个带选择相似度最高且满足几何约束的候选。
5. 面片几何从小、中尺度原始DFN面片继承，不生成大尺度裂缝。

## 几何约束

- 最小模板倾角：35°
- 补充片与控制点最小TWT偏移：3 ms
- 补充片与已有面片最小三维中心间距：12 m
- 每个井控事件最多补充三个带各一片
- 补充片比例：小尺度为主，中尺度次之，无大尺度

## 输出判别字段

- `IsNewConventionalControl`：新增常规测井井控事件
- `SeismicGuidedDensity`：地震属性引导的补充片
- `SeismicSimilarity`：候选与井控点的属性相似度
- `SeismicAttributeDistance`：归一化属性距离
- `SeismicCandidateBand`：候选所在径向带
- `TimeOffsetFromControlMs`：候选相对井控事件的TWT偏移
- `Minimum3DSeparationM`：候选与已有面片的最近三维距离
- `SeismicTemplatePatchID`：补充片继承的原始DFN模板

## 验收文件

- `well_control_correction_audit.csv`：井控事件与面片对应关系
- `seismic_guided_densification_audit.csv`：所有候选、相似度和拒绝原因
- `seismic_guided_densification_summary.json`：补充片汇总
- `well_corrected_dfn_summary.json`：Step8总QC
- `well_corrected_dfn_fracture_patches.csv`：最终面片主表
- `well_corrected_dfn_raw_time.vtk`：最终三维DFN

## 注意

常规测井模型不直接测量裂缝倾向倾角。新增井事件面片的倾向倾角继承同一层位、时窗附近的原始DFN模板；这是用于DFN井控和剖面表达的局部方向补全，不是成像测井实测产状。

## 本次验收结果

- Step8总状态：`pass`
- 37口井、525个控制点
- 新增井事件：35个
- 地震属性候选：840个
- 最终地震引导补充片：96个
  - 小尺度63个
  - 中尺度33个
  - 大尺度0个
- 井周200 m面片中心数：
  - 车152：8增加到84
  - 车斜155：45增加到93
- 真实DFN穿井交线：
  - 车152：39
  - 车斜155：30
- 所有补充片中心TWT偏移为5-8 ms，最小三维间距13.78 m。
