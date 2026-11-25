# 砂砾岩新增井裂缝预测（2026-11-25）

本文记录 `车152`、`车斜155` 两口新增常规测井井的外推预测入口。

## 输入

```text
/data/shared/project-oil/wx数据/砂砾岩/车镇--新井/
```

固定选用：

| 井 | 曲线 LAS | 井斜 | 电阻率对 |
|---|---|---|---|
| 车152 | `c152.las` | `c152_dev_all.txt` | `LLD_LLS` |
| 车斜155 | `chexie155_cur1.las` | `车斜155井井斜数据.csv` | `RD_RS` |

车斜155 的 `chexie155_cur1.las` 与 `车斜155-3.las` 共用同一测井网格和主要曲线值。
前者保留正确的密度空值，后者在约 3242-4000 m 将 4,967 个密度空值写为 0，
因此曲线主数据使用前者，井斜数据使用独立 CSV。

## 坐标

坐标文档使用测量习惯“纵 X / 横 Y”，与项目坐标相反，并且 Easting 带
20,000,000 前缀。

转换后：

```text
车152:   X=575760.42, Y=4202380.17
车斜155: X=575039.57, Y=4204709.89
```

## 轨迹与时深

- 井斜采用最小曲率法积分，计算 `MD/TVD/X/Y`。
- 时深不借用最近井的 MD/TIME 直接替换，而是使用来源井的 `TVD/TIME` 关系，
  代入目标井自身井斜积分得到的 TVD。
- 选择规则：距离最近，同时来源井 MD 和 TVD 范围完整覆盖目标井测井 MD/TVD 范围。
- 当前两口井最近且满足覆盖要求的来源均为 `车古271.dat`：
  - 车152 距来源约 606 m；
  - 车斜155 距来源约 3034 m。

## 模型

只加载现有模型，不重新训练：

```text
output/formal_multilog_expert_library_v2/model_library/
```

车152使用 `LLD_LLS` 专家；车斜155使用 `RD_RS` 专家。

## 当前结果

| 井 | 输入曲线对 | 六曲线完整行 | 进入目标层行 | 裂缝存在行 | 离散裂缝点 |
|---|---|---|---:|---:|---:|
| 车152 | `LLD_LLS` | 9,429 | 9,429 | 3,052 | 442 |
| 车斜155 | `RD_RS` | 7,571 | 6,765 | 411 | 99 |

分地层预测覆盖：

```text
车152 沙三段：3850.08-4469.89 m，8,135 行，383 点
车152 沙四段：4469.97-4568.50 m，1,294 行，59 点
车斜155 沙三段：3999.89-4748.48 m，4,913 行，41 点
车斜155 沙四段：4748.63-5030.72 m，1,852 行，58 点
```

车斜155 的井斜积分在校正后的 72 个参考测点范围内：

- 最大 TVD 偏差 0.039 m；
- 最大东西向偏差 0.254 m；
- 最大南北向偏差 0.183 m。

## 运行

```bash
python3 优化阶段二/正式主线/step4_expert_real_well_prediction/predict_new_wells_from_las.py \
  --config 优化阶段二/正式主线/step4_expert_real_well_prediction/configs/formal_new_well_inference_20261125.json \
  --replace-existing-output
```

## 输出

```text
output/formal_new_well_inference_20261125/
├── input_qc/
├── step2_like/
├── predictions/
├── provenance/
├── plots/
└── new_well_prediction_summary.json
```

主预测表：

```text
predictions/all_new_wells_merged_density_prediction.csv
predictions/all_new_wells_fracture_points.csv
```

逐井图：

```text
plots/车152_fracture_prediction.png
plots/车斜155_fracture_prediction.png
```

## 限制

新增井没有独立成像裂缝真值，因此只能评价覆盖、连续性和结果合理性，
不能直接用这两口井计算准确率、召回率或 AUC。

## 200 m DFN 剖面对比

入口：

```text
step9_section_visualize/build_new_well_dfn_prediction_sections.py
step9_section_visualize/configs/formal_new_wells_dfn_prediction_compare_20261125.json
```

该流程不读取成像测井标签，只比较：

- 常规测井最终细化裂缝点位 `all_new_wells_fracture_points.csv`，不合并；
- Step7D 井控前 DFN；
- Step8 井控后 DFN；
- 原始断层，单独显示。

每口井生成 XZ、YZ 两张井周约 200 m 剖面。DFN 按小、中、大尺度着色，
以地震波形和正振幅变面积作背景。常规测井细化点保留在真实 MD/TVD/TIME，
优先采用真实穿井 DFN 面片的产状；没有直接相交时，从同地层时空邻域 DFN
面片计算局部倾向倾角，并记录来源和置信度。1 m 内存在真实 DFN 穿井点时加匹配圈。

输出目录：

```text
step9_section_visualize/output/new_well_dfn_prediction_sections_20261125
```
