# 太古界 Step9b：分地层 DFN 裂缝密度量化

移植砂砾岩正式主线 `优化阶段二/正式主线/step9b_dfn_density_metrics/dfn_density_metrics.py`
（P10 / P21 / P33 设计见 `DFN密度量化指标设计_20260824.md`），把"沙三段 / 沙四段"
换成太古界的两套目标层，产出可直接汇报的**分地层裂缝密度图**与配套数据表。

## 1. 地层定义（与层位契约一致）

| 地层 | 顶 → 底 | 层位字段 | 全区平均厚度 |
|---|---|---|---|
| **上部复合层** | 上部复合层顶 → 太古界顶 | `T4 → T6` | 133.7 ms |
| **太古界风化壳** | 太古界顶 → 风化壳底 | `T6 → T7` | 75.9 ms |

来源：`太古界/common/horizon_contract/output/taigu_attribute_full_horizon_contract_v1/demo_horizon_contract.npy`
（表内存的是 `T4/T5/T6/T7`，其中 `T5 == T6`；层序有效性由 `SurfaceOrderValid` 标记，
全区有效道 1,892,935 / 2,156,550，约 87.8%）。

## 2. 指标口径（与砂砾岩一致）

### P33（体密度）—— 本步主指标

```
P33 = Σ(裂缝片面积 × 等效开度) / 体元体积
等效开度 a(mm) = a0 + (a1·factor − a0) · SourceDensity^0.5
```

| 尺度 | 开度区间 a0–a1（mm） |
|---|---|
| small | 0.1 – 1.0 |
| medium | 0.5 – 2.0 |
| large | 1.0 – 3.0 |

- 网格：**300 m × 300 m**（按道头 X/Y 分箱）；体元体积 = 网格面积 × 层段时间厚度；
- 单道平均面积用"外接矩形面积 ÷ 总道数"推得 = **156.06 m²**（等效道距 12.5 m）。
  ⚠ 不能直接对道头唯一 X/Y 取差分：OBN 道头坐标带 ±1 m 取整抖动，差分中位数会偏成 12×11；
- 层段厚度按网格内**层位有效道**取平均（无效道不计入，避免边缘 12% 缺层道把厚度拉低）；
- 开度敏感性输出 0.5 / 1 / 2 mm 三档，对外以 **1 mm 档** 为主；
- 单位：`m³/(m²·ms)`（时间域相对口径），图中放大 **1e6** 显示。

### P10（线密度）

井轨迹半径 150 m 内、落在该地层时间窗内的裂缝片数 ÷ 层段时间厚度 × 100（条/100 ms）。
层段时间窗取井位**邻域 25 条有效道**的 T4/T6/T7 中位数，避免单道噪声与边缘缺层道。

### P21（面密度）

砂砾岩版需要 Step9 剖面扫描结果，本版未产出（沿用砂砾岩 `dfn_density_metrics.py` 的预留状态）。

## 3. ⚠ 未做成像标定（与砂砾岩的唯一实质差异）

砂砾岩用 **车页1导眼** 的成像 P33 做"按层段常系数标定"，把 DFN 相对体密度抬到成像绝对量级
（沙三 14.15 / 沙四 51.74）。太古界**本版不做**这一步，原因：

1. 太古界成像井（埕北310 / 313 / 816、桩斜169、桩海102）井位在 **Y 4.226–4.232 M**，
   位于本次矿区网格（Y ≥ 4.2317 M）**南侧之外**，网格内没有可标定单元；
2. 网格内唯一的成像井埕北古斜405，其成像解释段（MD 3676.6–3998.0 → TIME 2746.8–2866.2）
   **整体落在目标层之上**，Step8 已记为"本区块没有可用成像井控"；
3. 太古界现有成像成果只有密度曲线与产状点，**没有现成的 P33 列**（砂砾岩的 P33 来自
   `exp_strata_ac_gr_v1/labeled_datasets/*_sample_labeled.csv`，属砂砾岩专有）。

因此本目录图件全部是 **DFN 相对体密度**，横向/纵向对比有效，绝对量级不等于成像 P33。
若后续给出太古界成像 P33 或井周成像密度→P33 的换算，可在脚本里加
`--calibration-csv / --calibration-well` 复用砂砾岩的标定代码路径。

## 4. 输入

| 输入 | 路径 |
|---|---|
| Step8 井控校正片表 | `太古界/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1/well_corrected_dfn_fracture_patches.csv` |
| 层位契约 | `太古界/common/horizon_contract/output/taigu_attribute_full_horizon_contract_v1/demo_horizon_contract.npy` |
| 道头（X/Y） | `太古界/common/attribute_trace_contract/output/taigu_attribute_trace_header_v1/attribute_trace_header.csv` |
| 井轨迹（P10） | `太古界/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1/well_trajectories_raw_time.vtk` |

参与统计的裂缝片 **370,617** 片：上部复合层 253,723（small 225,871 / medium 27,489 / large 363），
太古界风化壳 116,894（small 97,374 / medium 19,380 / large 140）。

## 5. 输出

目录：`太古界/step9b_dfn_density_metrics/output/taigu_quanqu1_v1/`

| 文件 | 内容 |
|---|---|
| `p33_heatmap_upper.png` | 上部复合层 P33 平面热图 |
| `p33_heatmap_crust.png` | 太古界风化壳 P33 平面热图 |
| `p33_heatmap_total.png` | 两层相加 P33 热图 |
| `p33_heatmap_{upper,crust}_unified.png` | 两层**统一色标**版（可直接互相对比） |
| `p33_heatmap_{upper,crust,total}_wells.png` | 井位 + 井名叠加版（等比例平面图） |
| `p33_scale_layer_bars.png` | 分尺度 × 分地层 P33 柱状图 |
| `p33_layer_compare.png` | 分地层体积平均 P33 对比 |
| `p33_grid_300m.csv` | 300 m 网格 × 地层：片数、裂缝面积、层段厚度、P33 三档开度 |
| `p33_summary_by_layer_scale.csv` | 地层 × 尺度汇总（片数/面积/等效体积三档） |
| `p10_well_summary.csv` | 11 口井 × 地层 × 尺度 P10 |
| `dfn_density_metrics_metadata.json` | 口径、输入、计数与未标定说明 |

方向约定与砂砾岩一致：**平面俯视，X 自左向右增大，Y 自上向下减小（北在上）**。

## 6. 运行

```bash
# 主计算（默认 python 无 vtk，用 gan-dfn 环境）
/home/tyh/anaconda3/envs/gan-dfn/bin/python3 太古界/step9b_dfn_density_metrics/dfn_density_metrics.py

# 井位叠加图
/home/tyh/anaconda3/envs/gan-dfn/bin/python3 太古界/step9b_dfn_density_metrics/plot_wells_on_density.py \
  --grid 太古界/step9b_dfn_density_metrics/output/taigu_quanqu1_v1/p33_grid_300m.csv \
  --wells 太古界/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1/well_trajectories_annotations.csv \
  --vtk 太古界/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1/well_trajectories_raw_time.vtk \
  --out-dir 太古界/step9b_dfn_density_metrics/output/taigu_quanqu1_v1
```

## 7. 展示建议与已知局限

- **数量讲构成、密度讲强弱**：片数/面积讲构成，P33 讲发育程度；两张分地层图用 `_unified`
  版并排，差异一眼可见；
- 热图是 300 m 网格斑点状分布，属 DFN 空间不均匀的固有表现，不是插值瑕疵；
- 网格边缘（南侧、东北角）为缺层位 / 缺裂缝片区，图中为 0，解读时不要当作"真零密度"；
- 等效开度是幂律假设、非实测；时间域 P33 为相对口径，转物理孔隙度需要层速度；
- 与砂砾岩对比时注意两边**独立**：地层不同、网格相同（300 m）、开度映射相同，但太古界未做成像标定。
