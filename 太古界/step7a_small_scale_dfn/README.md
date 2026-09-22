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
