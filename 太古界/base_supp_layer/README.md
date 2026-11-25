# 太古界基底以下补充层试点（5×5 km）

## 目的

沿用“上部补充层”的局部重跑思路，在太古界风化壳底以下增加 300 ms
低置信度 DFN，并验证它与现有正式 DFN 的拼接和时间连续性。

当前只处理原上部补充层的 401×401 试点区：

```text
X 663450–668450
Y 4239250–4244250
面积 25 km²
```

## 时间与层位口径

逐道窗口：

```text
Top = 风化壳底 Base
Bottom = min(Base + 300 ms, 4800 ms)
```

蚂蚁体、相干体、曲率体的绝对时间范围均为 1800–4800 ms。
试点区 160,801 道中有 160,520 道有效，全部可以完整下拓 300 ms。

新增层名和编码：

```text
LayerGroup = 基底以下补充层
LayerCode  = 4
```

基底以下当前没有目标层真实裂缝标签。因此 Step6A 密度使用
“太古界风化壳”两阶段模型作为代理，所有结果属于低置信度属性外推。

## 流程

1. 生成逐道 `Base–BaseV` 合同和局部网格。
2. 在 2 ms 时间轴上生成代理密度体。
3. 重跑 Step6A/6B/6C/6D。
4. 重跑 Step7A/7B/7C/7D。
5. 把临时 T4/T6/T7 层槽位统一重标签为“基底以下补充层”。
6. 追加到当前 `seg1` 正式 DFN，并验证边界连续性。

完整重跑：

```bash
bash 太古界/base_supp_layer/run_base_supp_5km_pilot.sh
```

## 主要输出

根目录：

```text
太古界/base_supp_layer/output/taigu_base_supp_5km_v1
```

单独的下部补充 DFN：

```text
merged/base_supp_dfn_fracture_patches.csv
merged/base_supp_dfn_predicted_raw_time.vtk
```

与正式 `seg1` DFN 的拼接结果：

```text
merged/merged_main_plus_base_supp_patches.csv
merged/merged_predicted_dfn_raw_time.vtk
merged/merged_unified_dfn_raw_time.vtk
```

边界连续性和总 QC：

```text
merged/base_supp_boundary_continuity.csv
merged/base_supp_merge_qc_summary.json
```

## 当前验证结果

| 项目 | 结果 |
|---|---|
| 下部补充 DFN | 46,869 片 |
| 小 / 中 / 大尺度 | 41,916 / 4,473 / 480 |
| 下部中心相对 Base | 0.0–299.9 ms |
| 主 DFN | 371,319 片 |
| 拼接后预测片 | 418,188 片 |
| 原始断层三角面 | 41,713，只保留一份 |
| 拼接后 VTK | 459,901 单元 |
| 基底上下 100 ms 最大中心空带 | 0.1 ms |
| 试点区内主 DFN 越过 Base | 0 片 |
| Step6/7 状态 | 全部 pass |
| 拼接 QC | pass |

VTK 字段：

- `GeometryGroupCode=1`：预测片。
- `GeometryGroupCode=2`：原始断层三角面。
- `IsOriginalFault=0`：预测片。
- `IsOriginalFault=1`：原始断层。
- `LayerCode=4`：基底以下补充层。
- `IsBaseSupplement=1`：基底以下补充片。

## 注意事项

- 没有对基底以下做无标签重新训练。
- 没有对 Step8 井控进行伪造；本次先验证 Step6/7 DFN、拼接和时间连续性。
- 主 DFN 全区有 6 片中尺度中心低于各自 Base 1.8–34.6 ms，均位于试点区外；
  试点区内为 0 片。
- 面片跨越 Base 属于正常地质连续现象。拼接按中心时间划分，不切平面片。

## 上部补充层 + 原始 DFN + 下部补充层

三路合并脚本：

```bash
python3 太古界/base_supp_layer/merge_three_way_layers.py
```

输入：

```text
原始 DFN：taigu_step8_attribute_multiscale_v1_quanqu1_seg1
上部补充层：taigu_supp_step8_v1
下部补充层：taigu_base_supp_5km_v1
```

输出：

```text
太古界/base_supp_layer/output/taigu_three_way_merge_v1/
├── three_way_merged_dfn_patches.csv
├── three_way_predicted_dfn_raw_time.vtk
├── three_way_unified_dfn_raw_time.vtk
└── three_way_merge_summary.json
```

三路合并结果：

| 项目 | 数量 |
|---|---:|
| 原始 DFN | 371,319 |
| 上部补充层 | 47,407 |
| 下部补充层 | 46,869 |
| 合并预测片 | 465,595 |
| 原始断层三角面 | 41,713 |
| 统一 VTK 单元 | 507,308 |

VTK 字段：

- `LayerCode=1/2`：原始 DFN 的上部复合层 / 太古界风化壳。
- `LayerCode=3`：上部补充层。
- `LayerCode=4`：基底以下补充层。
- `IsUpperSupplement=1`：上部补充片。
- `IsBaseSupplement=1`：基底以下补充片。
- `IsOriginalFault=1`：原始断层三角面。

## 5 km 三路合并剖面

配置：

```text
太古界/base_supp_layer/configs/taigu_three_way_5km_sections.json
```

输出：

```text
太古界/base_supp_layer/output/taigu_three_way_merge_v1/5km_sections
```

生成 10 张 XZ/YZ 分尺度剖面和 3 张独立图例：

```text
01_large_xz_t4_t7.png ... 10_all_scales_yz_t4_t7.png
legend_01_large.png
legend_02_large_medium.png
legend_03_all_scales.png
```

剖面范围：

```text
X 663450–668450
Y 4239262–4244262
显示时间 2534–4268 ms
```

剖面中上部补充层并入“上部复合层”显示，下部补充层并入
“太古界风化壳”显示；裂缝颜色仍严格按小、中、大尺度区分。

完整重跑：

```bash
bash 太古界/base_supp_layer/run_three_way_5km.sh
```

注意：部分下部小/中尺度片没有 `V1Z–V4Z` 顶点字段。剖面时间范围
现在对这类片使用 `CenterTime ± HeightTimeMs/2` 回退，避免漏掉下部延伸段。

## 7. ext300_600_v2 小范围验证（2026-11-25）

新版验证区仍为 405 井周 5×5 km，但窗口改为：

```text
上拓：max(Top - 300, 1800) -> Top
下拓：Base -> min(Base + 600, 4800)
```

使用 `taigu_step6a_attribute_multilog_v2` 模型，输出分为：

```text
太古界/supp_layer/output/taigu_supp_pilot_ext300_v2
太古界/base_supp_layer/output/taigu_base_supp_pilot600_v2
太古界/base_supp_layer/output/taigu_three_way_pilot_ext300_600_v2
```

结果：

- 上拓 Step6 全部 pass；Step7B pass，4,078 片；Step7D pass，48,511 片；
  Step8 pass，50,653 片。
- 上拓 300 ms 对 405 成像段的覆盖为 2,558/2,572（99.46%）；14 点仅超出
  TopV 约 0.66 ms。
- 上拓 7B 已改为读取新 Step6B 先验并显式使用 5×5 目标块，验收 pass。
  上拓 7C 读取新 Step6C 的 `2200–3700 ms` 时间轴，按上部单层语义检查，
  验收 pass，194 片。
- 上拓 7C 保留 1 个中心道缺层位、但裂缝面体素覆盖率满足要求的跨边界片：
  `large_inferred_fault_surface_00056`（`CenterInWindow=0`，
  `WindowVoxelFraction=1.0`）；三路合并后编号为
  `upper_supp_fused_multiscale_000049`。
- 下拓 Step6/7/合并全部 pass，84,314 片；中心相对 Base 为 0–599.9 ms。
- Base 上下 100 ms 最大中心空带 0.1 ms。
- 三路合并 pass：预测片 506,286，四类层码完整，原始断层只保留一份。
- 统一摘要见 `taigu_three_way_pilot_ext300_600_v2/validation_summary.json`。
