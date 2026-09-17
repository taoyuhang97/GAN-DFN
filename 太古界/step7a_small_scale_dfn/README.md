# Step7A 太古界小尺度 DFN

从三维小尺度裂缝密度体出发，按"分层候选 + 密度加权概率采样 + 三维最小间距"生成裂缝片几何，
产状由成像测井统计的裂缝组（family）加权、并按局部梯度脊做微调。

## 输入源（2026-09-17 起）

配置：`configs/taigu_step7a_attribute_v3_regen_v4.json`

| 输入 | 路径 |
|---|---|
| 小尺度密度体 | `step6d_multiscale_bundle/output/taigu_step6d_multiscale_v4/final_small_density.sgy` |
| 输入契约摘要 | `.../taigu_step6d_multiscale_v4/multiscale_bundle_summary.json` |
| 层位合同 | `common/horizon_contract/output/taigu_attribute_full_horizon_contract_v1/demo_horizon_contract.csv` |
| 成像监督 | `step3_imaging_groups/output/taigu_step3_imaging_v4/groups` |

说明：原先直接消费 Step6A 的模型密度体，导致"模型没学到的属性关系"没有兜底；
现改为消费 Step6D 的 `final_small_density.sgy`（含 Step6A 0.4/0.6 小尺度背景融合与中/大尺度损伤带派生片）。

## 关键验收项

- `all_patches_main_window`：所有片的主窗归属（延伸窗尚未进入 DFN，见问题记录 P0-7）；
- `minimum_3d_spacing_respected`：三维最小间距；
- `orientation_point_join_has_no_unmatched`：Step3 成像点回接（段池 + 容差）无未命中；
- `both_layers_represented`：上部复合层与风化壳都有片。

输出 `fracture_patches.csv` / `.vtk`、`orientation_families.csv`、`step7a_sampling_qc.json`、`step7a_summary.json`。
