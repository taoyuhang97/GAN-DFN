# Step7A 太古界小尺度 DFN

从三维小尺度裂缝密度体出发，按"分层候选 + 密度加权概率采样 + 三维最小间距"生成裂缝片几何，
产状由成像测井统计的裂缝组（family）加权、并按局部梯度脊做微调。

## 输入源（v5，2026-09-17 起：分域采样）

配置：`configs/taigu_step7a_attribute_v3_regen_v5.json`

| 域 | 输入体 | 口径 |
|---|---|---|
| 背景（主体） | Step6D `background_small_density.sgy` | 0.06 + 0.85×Step6A 小尺度背景；候选 q0.9、出现率 0.015/0.01、幂 1.5 |
| 断层派生（受限） | Step6D `large_damage_small_density.sgy` | 断层损伤壳；候选 q0.5、出现率 0.005、幂 1.0；总量 ≤0.75×背景片数 |

| 其他输入 | 路径 |
|---|---|
| 输入契约摘要 | `.../taigu_step6d_multiscale_v4/multiscale_bundle_summary.json` |
| 层位合同 | `common/horizon_contract/output/taigu_attribute_full_horizon_contract_v1/demo_horizon_contract.csv` |
| 成像监督 | `step3_imaging_groups/output/taigu_step3_imaging_v4/groups` |

历史 v4（已废弃）：直接把 Step6D 的 `final_small_density`（背景 / 中尺度损伤 / 断层损伤
三者取最大值）当作单一输入，导致断层损伤带的 0.5 平台整片进入候选集、小尺度 DFN 变成断层形状
（详见 `太古界/太古界流程梳理与问题记录_20260915.md` 0.25 节）。

## 输出列（v5 新增）

`fracture_patches.csv` 新增 `SmallDomain`（background / large_fault_damage）、
`SmallDomainCode`（1 / 2）、`StructuralRelation`、`SamplingPriority`，
用于对外区分"背景裂缝"与"断层旁派生小裂缝"。

## 关键验收项

- `all_patches_main_window`：所有片的主窗归属（延伸窗尚未进入 DFN，见问题记录 P0-7）；
- `minimum_3d_spacing_respected`：三维最小间距；
- `orientation_point_join_has_no_unmatched`：Step3 成像点回接（段池 + 容差）无未命中；
- `both_layers_represented`：上部复合层与风化壳都有片；
- `damage_domain_within_configured_cap`：断层派生片不超过 0.75×背景片数；
- `background_domain_is_majority`：背景域片数多于断层派生域。

输出 `fracture_patches.csv` / `.vtk`、`orientation_families.csv`、`step7a_sampling_qc.json`、`step7a_summary.json`。


---

## 口径变更（2026-09-22，azi1）

产状统一为 **真倾向方位 0–360（自北顺时针）+ 倾角**，唯一实现见
`太古界/common/orientation_frame/convention.py`（换算规则与验证见该目录 README）。

- `DipAzimuthDeg` = 唯一真值字段；`AzimuthDeg` 降级为派生走向 `(DipAzimuthDeg-90)%180`。
- 第三维为 `TIME`（向下为正）的帧一律走 `*_depth` 变体换算。
- 旧口径开关（`family_azimuth_semantics` / `ridge_azimuth_semantics` /
  `vertex_convention` / `dfn_azimuth_semantics`）已删除，不再保留双口径分支。
- 背景：`太古界/太古界流程梳理与问题记录_20260915.md` §0.37；
  施工方案：`太古界/太古界产状口径统一施工方案_20260922.md`。


---

## 产状赋值逻辑变更（2026-09-23，azi4）

小尺度片的产状不再由"多井家族 KMeans + 均值"给出，改为
**按井经验分布重采样 + 局部几何 β 门控**：

1. 按 `(层, 井)` 保留原始 `(倾向方位, 倾角)` 经验库（不聚类、不平均），库内包含**全部**成像点；
2. 用**高斯距离核**给最近的几口井加权（`well_weight_scale_m`），权重随位置处处连续；
3. 用**空间相关随机场**（`well_assignment_coherence_m`）抽井，保证相邻片倾向同一口井；
4. 从抽中井的经验库重采样并加小幅抖动（`azimuth_jitter_std_deg` / `dip_jitter_std_deg`）；
5. **局部地震几何只在走向差 ≤ `local_geometry_strike_tolerance_deg` 时**才允许以
   ≤ `local_geometry_dip_blend_max` 的权重融合倾角，否则完全忽略。

输出文件由 `orientation_families.json/csv` 改为 `orientation_well_library.json/csv`（按井审计表）；
片表新增审计列 `OrientationWellDraw` / `OrientationWellWeights` / `OrientationEffectiveWellCount` /
`LocalGeometryUsed` / `LocalGeometryBlendWeight` / `LocalGeometryStrikeDeltaDeg`。

方案与实测：`太古界/太古界DFN产状赋值逻辑修正方案_20260923.md`；
问题记录 §0.38。本轮**只改小尺度（7A）**，中尺度按决策 D4 保持蚂蚁体来源不变。
