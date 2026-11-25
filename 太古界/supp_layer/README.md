# 太古界上部补充层（405 局部 5×5 km）

目的：在太古界现有"上部复合层 / 太古界风化壳"之上，再补一层**上部补充层**，
把埕北古斜405的成像解释段包进来，使 Step9 / Step9b 能出可与现有两层对比的剖面图。

> 约束：**不修改原代码**。本目录只新增配置与脚本；新框架在运行时注册进
> `common/strata_framework/registry.py` 的 `REGISTRY` 字典（不改文件）。

## 1. 层序

```
TopV（虚拟顶，新增）
  └─ 上部补充层（code=3）
Top（上部复合层顶，现有）
  └─ 上部复合层（code=1）
Mid（太古界顶，现有）
  └─ 太古界风化壳（code=2，底界以下含 50 ms 延伸）
Base（风化壳底，现有）
```

太古界只有 Top / Mid / Base 三个真实层位面，不存在 T4–T7；本目录一律使用这套命名。

## 2. 虚拟顶

```
TopV(trace) = max( TopTimeMs(trace) − 350 ms , 2650 ms )
```

| 参数 | 取值 | 理由 |
|---|---|---|
| 上抬量 | 350 ms | 405 成像段处 Top 比成像时间高 125.2–300.7 ms（中位 222.1），350 留约 49 ms 余量 |
| 下限 | 2650 ms | 6B/6C 相干/蚂蚁/曲率属性体从 2624 ms 起，留 26 ms 余量 |

在 405 处：TopV = **2650–2698 ms**，成像段 2746.8–2866.2 ms 完整落入新层，
最小余量 **49.3 ms**。

## 3. 局部区块

以 405 轨迹中心（X 665,954 / Y 4,241,759）为中心的 5×5 km：

```
X 663,450 – 668,450
Y 4,239,250 – 4,244,250
```

区块内 160,682 道，其中 159,140 道能形成有效上部补充层（99.0%）。

## 4. 目录内容

```text
太古界/supp_layer/
├── configs/taigu_supp_v1.json        # 新地层框架（TopV/Top/Mid/Base + 3 层）
├── supp_framework.py                 # 运行时注册 taigu_supp_v1
├── build_supp_horizon.py             # 生成带 TopV 的逐道契约
├── build_supp_samples.py             # 405 成像监督组 + 常规测井样本
├── predict_supp_density.py           # Step6A：区块内新层密度预测（复用上部复合层模型）
├── build_supp_legacy_adapter.py      # 把新层映射成 T4/T6/T7 形状，供 6B/6C/6D 复用
├── build_supp_step7_inputs.py        # Step7 区块网格 / window_code / 四份配置
├── relabel_supp_dfn.py               # 把 7D 临时层标签改回"上部补充层"（LayerCode=3）
├── build_supp_step8_inputs.py        # Step8 临时层标签成像组 + 配置
├── merge_supp_dfn.py                 # 新层 DFN 与现有全域 DFN 几何并集
├── plot_supp_layer_density.py        # Step9b：5×5 km 三层裂缝面积密度对比
├── plot_supp_step9_split_sections.py # Step9 增补：成像真值 / DFN 预测拆成两张剖面
├── plot_supp_step9_wiggle_split.py   # Step9 增补：地震波形+变面积底图上的成像 / DFN 拆分剖面
├── plot_supp_step9_scale_wiggle.py   # Step9 增补：矿区尺度波形+变面积的分尺度 DFN 剖面
├── configs/taigu_supp_step9_wiggle_split.json
├── configs/taigu_supp_step9_scale_wiggle.json
├── README.md
└── output/                           # 生成物（已 gitignore）
    ├── taigu_supp_horizon_v1/
    │   ├── supp_horizon_contract.npy          # 全矿区 2,156,550 道
    │   ├── supp_horizon_contract_block.csv    # 区块 160,682 道
    │   └── supp_horizon_summary.json
    ├── taigu_supp_samples_v1/
        ├── supp_imaging_group.csv             # Step3 组格式
        ├── supp_real_well_samples.csv         # 带 GR/RD/RS/Density 的井上样本
        └── supp_samples_summary.json
    ├── taigu_supp_density_v1/
        ├── supp_density_block.sgy             # 159,140 道 × 493 样点 @2 ms
        ├── supp_density_mapping.npz
        └── supp_density_summary.json
    ├── taigu_supp_legacy_v1/
        ├── supp_horizon_legacy_full.csv/.npy  # T4=TopV, T6=Top-1ms, T7=Top
        ├── supp_6b_config.json / supp_6c_config.json / supp_6d_config.json
        ├── step6b_supp/                        # 中尺度先验
        ├── step6c_supp/                        # 大尺度断层先验
        └── step6d_supp/                        # 多尺度打包
    ├── taigu_supp_step7_v1/
        ├── supp_demo_grid_block.csv            # 401×401 区块网格（局部 IX/IY）
        ├── supp_window_code.npz                # 新层层窗（160801×494）
        ├── supp_7a_config.json ... supp_7d_config.json
        ├── step7a_supp/                        # 小尺度 43,061 片
        ├── step7b_supp/                        # 中尺度 3,919 片
        ├── step7c_supp/                        # 大尺度 159 片
        └── step7d_supp/                        # 融合 47,139 片 + 修正层标签 CSV
    └── taigu_supp_step8_v1/
        ├── supp_8_config.json
        ├── groups/                             # 临时改名为"上部复合层"的 405 成像组
        └── step8_supp/                         # 井控校正 47,407 片 + 修正层标签 CSV
    └── taigu_supp_merged_v1/
        ├── merged_dfn_patches.csv              # 418,024 片（三层）
        ├── merged_dfn_raw_time.vtk             # 462,976 单元
        └── merged_dfn_summary.json
    └── taigu_supp_step9_v1/               # 剖面（20 图全量 + 2 图快速验证）
    └── taigu_supp_step9b_v1/
        ├── layer_area_density_grid_300m.csv
        └── layer_area_density_compare.png
    └── taigu_supp_step9_split_v1/
        ├── 01_imaging_trajectory_and_fractures.png   # 仅井轨迹 + 真实裂缝
        └── 02_dfn_predicted_only.png                 # 仅 DFN 预测
    └── taigu_supp_step9_wiggle_split_v1/
        ├── 01_wiggle_area_imaging.png                # 地震底图 + 成像轨迹/真实裂缝
        └── 02_wiggle_area_dfn.png                    # 地震底图 + 成像段井轨迹 + DFN
    └── taigu_supp_step9_scale_wiggle_v1/
        ├── 01_large_xz_t4_t7.png ... 10_all_scales_yz_t4_t7.png
        └── legend_01_large.png ... legend_03_all_scales.svg
```

## 5. 运行

```bash
cd /home/tyh/projects/petroleum/code/GAN-DFN

# 1) 逐道契约（含 TopV）
python3 太古界/supp_layer/build_supp_horizon.py

# 2) 405 成像组 + 井上样本
python3 太古界/supp_layer/build_supp_samples.py

# 3) Step6A：区块内新层密度体（复用上部复合层模型）
python3 太古界/supp_layer/predict_supp_density.py

# 4) 兼容适配 + Step6B/C/D（复用原有脚本，不改代码）
python3 太古界/supp_layer/build_supp_legacy_adapter.py
python3 太古界/step6b_medium_corridor/code/build_step6b_medium_corridor_prior.py \
  --config 太古界/supp_layer/output/taigu_supp_legacy_v1/supp_6b_config.json \
  --output-dir 太古界/supp_layer/output/taigu_supp_legacy_v1/step6b_supp
python3 太古界/step6c_large_fault/code/build_step6c_large_fault_prior.py \
  --config 太古界/supp_layer/output/taigu_supp_legacy_v1/supp_6c_config.json
python3 太古界/step6a_density_volume/build_taigu_small_background.py \
  --config 太古界/supp_layer/output/taigu_supp_legacy_v1/supp_6a_small_config.json
python3 太古界/step6d_multiscale_bundle/code/build_step6d_multiscale_bundle.py \
  --config 太古界/supp_layer/output/taigu_supp_legacy_v1/supp_6d_config.json
```

两个脚本都可 `--help` 查看参数。

波形+变面积拆图可单独重跑（优先复用 Step9 的局部 OBN 采样缓存）：

```bash
python3 太古界/supp_layer/plot_supp_step9_wiggle_split.py \
  --config 太古界/supp_layer/configs/taigu_supp_step9_wiggle_split.json
```

右上大裂缝周边的定向小裂缝加密由独立后处理完成，不重新移动已条件化的
井周裂缝。当前 `v2` 以 `staged_v4` 的局部条件化交线为输入，对
`SourceIndex=371350` 大裂缝周边新增 256 条小尺度裂缝扰动副本：

```bash
python3 太古界/supp_layer/enrich_supp_small_patches_near_large.py \
  --input-csv 太古界/supp_layer/output/taigu_supp_step9_imaging_dfn_200m_staged_v4/local_conditioned_dfn_patches.csv \
  --output-csv 太古界/supp_layer/output/taigu_supp_step9_imaging_dfn_200m_staged_v4_style_v2_upper_right_enriched/upper_right_large_enriched_source_patches.csv \
  --summary-json 太古界/supp_layer/output/taigu_supp_step9_imaging_dfn_200m_staged_v4_style_v2_upper_right_enriched/upper_right_large_enrichment_summary.json \
  --anchor-source-index 371350 \
  --added-count 256
```

随后将 `upper_right_large_enriched_source_patches.csv` 作为
`plot_supp_step9_wiggle_split.py --dfn-csv` 的输入，并关闭二次大裂缝锚定
条件化。该加密只增加小尺度副本，不改动成像锁定裂缝、井控裂缝、中尺度
和大尺度裂缝。

若上半成像段的井上裂缝仍明显密于相交大裂缝周边，可继续以 `v2`
加密结果为输入，并限定 `2746–2808 ms` 做第二级局部加密：

```bash
python3 太古界/supp_layer/enrich_supp_small_patches_near_large.py \
  --input-csv 太古界/supp_layer/output/taigu_supp_step9_imaging_dfn_200m_staged_v4_style_v2_upper_right_enriched/upper_right_large_enriched_source_patches.csv \
  --output-csv 太古界/supp_layer/output/taigu_supp_step9_imaging_dfn_200m_staged_v4_style_v3_upper_segment_enriched/upper_segment_large_enriched_source_patches.csv \
  --summary-json 太古界/supp_layer/output/taigu_supp_step9_imaging_dfn_200m_staged_v4_style_v3_upper_segment_enriched/upper_segment_large_enrichment_summary.json \
  --anchor-source-index 371350 \
  --added-count 180 \
  --enrichment-stage upper_imaging_segment_large_fracture_enrichment_v3 \
  --time-min-ms 2746 --time-max-ms 2808 \
  --lateral-sigma-m 22 --max-lateral-offset-m 45 \
  --longitudinal-sigma-m 5 --max-longitudinal-offset-m 10 \
  --azimuth-sigma-deg 2 --dip-sigma-deg 2 --seed 406
```

`v4` 进一步平滑右上方大裂缝沿纵深方向的小裂缝密度。该步骤保持裂缝总数、
倾角、倾向和到大裂缝的水平偏移不变，仅重排沿 TIME 方向的位置：

```bash
python3 太古界/supp_layer/smooth_small_patches_along_large.py \
  --input-csv 太古界/supp_layer/output/taigu_supp_step9_imaging_dfn_200m_staged_v4_style_v3_upper_segment_enriched/upper_segment_large_enriched_source_patches.csv \
  --output-csv 太古界/supp_layer/output/taigu_supp_step9_imaging_dfn_200m_staged_v4_style_v4_wake_smoothed/wake_smoothed_source_patches.csv \
  --summary-json 太古界/supp_layer/output/taigu_supp_step9_imaging_dfn_200m_staged_v4_style_v4_wake_smoothed/wake_smoothing_summary.json \
  --anchor-source-index 371350 \
  --time-min-ms 2744 --time-max-ms 2862 \
  --smoothing-strength 0.85 --bin-width-ms 10 --seed 20261122
```

若上半成像段仍表现为“井上密、大裂缝周围稀”，可继续把对应小裂缝的水平
横向偏移压缩到 55%。该步骤保持 TIME 和产状不变，并逐条检查井轨迹，
恢复任何会新增井交点的小裂缝位置：

```bash
python3 太古界/supp_layer/concentrate_small_patches_near_large.py \
  --input-csv 太古界/supp_layer/output/taigu_supp_step9_imaging_dfn_200m_staged_v4_style_v4_wake_smoothed/wake_smoothed_source_patches.csv \
  --output-csv 太古界/supp_layer/output/taigu_supp_step9_imaging_dfn_200m_staged_v4_style_v5_upper_wake_concentrated/upper_wake_concentrated_source_patches.csv \
  --summary-json 太古界/supp_layer/output/taigu_supp_step9_imaging_dfn_200m_staged_v4_style_v5_upper_wake_concentrated/upper_wake_concentration_summary.json \
  --anchor-source-index 371350 \
  --time-min-ms 2744 --time-max-ms 2812 \
  --lateral-scale-factor 0.55 \
  --well-track-dir 太古界/step2_well_log_segments/output/taigu_step2_regular_md1 \
  --well-name 埕北古斜405
```

矿区分尺度波形剖面可单独重跑：

```bash
python3 太古界/supp_layer/plot_supp_step9_scale_wiggle.py \
  --config 太古界/supp_layer/configs/taigu_supp_step9_scale_wiggle.json
```

砂砾岩和太古界的独立图例由同一脚本生成。太古界“裂缝尺度”分组直接复用
`taigu_supp_step9_scale_wiggle.json` 中的 `scale_style`，用青、橙、红三种颜色
分别表示小、中、大尺度裂缝，并保留线宽作为辅助编码；不再按上部复合层、
风化壳和补充层列“地层裂缝”：

```bash
python3 scripts/build_step9_standalone_legends.py
```

## 6. 当前验证结果

| 项 | 结果 |
|---|---|
| 契约道数 | 2,156,550（TraceIdx 对齐） |
| 区块道数 | 160,682 |
| 区块有效补充层道 | 159,140（99.0%） |
| 405 成像点落入新层 | **2572 / 2572 = 100%** |
| 405 成像点余量 | 最小 49.3 ms / 中位 127.9 ms |
| 405 井上样本 | 2,573 行，GR/RD/RS 全有效 |
| 其中密度 > 0 | 1,829 行（71.1%），均值 0.98，最大 6.76 |
| **Step6A 密度体** | 159,140 道 × 493 样点（2650–3634 ms @2 ms），耗时 203 s |
| 405 成像点密度 | 2572/2572 有值，均值 0.892，最大 1.477 |
| **Step6B 中尺度** | status=pass，889 个组件，候选体素 391,325，层内占比 28.1% |
| **Step6C 大尺度** | status=pass，188 个推断组件，323 个原始断层片 |
| **Step6D 打包** | status=pass，shape 160,801×494（2650–3636 ms）；small 2.65M / medium 644k / large 1.31M 候选体素 |
| **Step7A 小尺度** | status=pass，43,061 片 | 75 s |
| **Step7B 中尺度** | 3,919 片；仅"空间四象限覆盖"QC 未过（局部区块候选集中） |
| **Step7C 大尺度** | 159 片；仅"双层代表"QC 未过（单层运行属预期） |
| **Step7D 融合** | **status=pass，47,139 片**（small 43,061 / medium 3,919 / large 159），时间 2650–3600 ms |
| 层标签修正 | 47,139 行全部改为 `上部补充层` / `LayerCode=3` |
| **Step8 井控校正** | **status=pass**，**301 个控制行（317 条真实裂缝）**，**47,407 片**（small 43,329 / medium 3,919 / large 159），仅修改 301 片，时间 2650–3600 ms |
| Step8 层标签修正 | 47,407 行全部改为 `上部补充层` / `LayerCode=3` |
| **拼接（新层 ∪ 全域）** | **418,024 片**（原 370,617 + 新 47,407）；VTK **462,976 单元 / 1,724,789 点**；三层：上部复合层 253,723 / 太古界风化壳 116,894 / 上部补充层 47,407 |
| **Step9 剖面** | 全量 20 图（2354–4232 ms，含新层纵轴范围）；另出 2 图快速版，临时把新层并入上部复合层以命中原配色表 |
| **Step9b 密度对比** | 5×5 km 300 m 网格面积密度：上部复合层均值 1.045、太古界风化壳 0.511、上部补充层 **1.779**（patch 27,157 / 12,110 / 47,278） |
| **Step9 拆分剖面** | 405 井段 ±200 m、2600–3060 ms：图1 仅井轨迹+301 行真实裂缝（317 条，按倾向/倾角画线）；图2 仅 DFN 预测（不含真实成像裂缝），按小/中/大尺度着色 |
| **Step9 波形+变面积拆分** | 同一窗口：波形+变面积底图；两张图统一使用原 Step9 的黄色轨迹“Step3成像井”图例标识；DFN 直接使用 `DipAzimuthDeg/DipDeg` 求剖面交线，不使用历史 VTK `V1..V4` 顶点；图例置于画布下方 |
| **Step9 矿区分尺度波形** | `status=pass`，5 套组合共 10 图；颜色和线宽按大/中/小尺度统一，图为 XZ/YZ 剖面 |
| **Step9 独立图例** | `status=pass`，3 套 PNG/SVG；太古界首列为“裂缝尺度”，用青、橙、红分别表示小、中、大尺度，旧三层裂缝标签和灰度图例均已移除 |
| **Step9 右上大裂缝定向加密 v2** | `status=pass`；`SourceIndex=371350` 周边小裂缝 818→1074，三维源数据新增 256 条；XZ 选入交线 2398→2645，中/大尺度数量不变 |
| **Step9 上半成像段定向加密 v3** | `status=pass`；`2746–2808 ms` 内大裂缝横向 30 m 小裂缝 126→178、45 m 内 167→237；XZ 选入交线 2645→2824，中/大尺度数量不变 |
| **Step9 右上大裂缝密度平滑 v4** | `status=pass`；`2744–2862 ms` 小裂缝总数保持 1048 条；10 ms 分箱数量波动系数 0.269→0.089 |
| **Step9 上半段大裂缝周边集中 v5** | `status=pass`；`2744–2812 ms` 内 30 m 横向带 161→239、45 m 带 214→310；未新增井交点，总裂缝数不变 |

## 7. 与原代码的接口

- 新框架通过 `supp_framework.load_supp_framework()` 注册后，
  `common/horizon_trace_table/horizon_contract.py` 即可按 `taigu_supp_v1`
  读取 TopV/Top/Mid/Base，**无需改代码**；
- `build_supp_horizon.py` 直接调用 `load_framework_file()`，不依赖注册表；
- 现有 `demo_horizon_contract.npy`、现有 DFN、现有 Step9/9b 产物**均未改动**。

## 8. 后续步骤（尚未实现）


> Step6A 已完成：`predict_supp_density.py` 复用"上部复合层"两阶段模型，
> 属性分数使用其 normalization contract；层窗取 `[TopV, Top)`，与上部复合层不重叠。

> Step6B/C/D 已完成：`build_supp_legacy_adapter.py` 把新层映射到 T4/T6/T7 槽位
> （T4=TopV、T6=Top−1 ms、T7=Top），原有 6B/6C/6D 脚本无需修改即可运行；
> 输出的 LayerGroup 仍是临时"沙三段"语义，Step7 之后会统一改回"上部补充层"。

> Step7A–D 已完成：7A/7B/7C 复用原有脚本（新区块配置）；7D status=pass，
> 融合 47,139 片。`relabel_supp_dfn.py` 已把层标签统一为"上部补充层"/LayerCode=3。

> Step8 已完成：`build_supp_step8_inputs.py` 直接使用**原始 405 Step3 成像组**
> （覆盖 MD 3676.55–3998.05，带真实 GT_POINT_FLAG/RawPointCount），只把层名临时
> 改成"上部复合层"以通过原脚本层过滤。真实裂缝共 **317 条**（CCF 70 + DCF 196 +
> ECF 51），分布在 **301 个采样行**；井控校正后 status=pass，47,407 片，仅修改
> 301 片，再用 `relabel_supp_dfn.py` 统一改回"上部补充层"/LayerCode=3。
>
> 早期版本曾用"密度>0 的 1,829 个采样行"作代理控制，导致井周补片偏多
> （48,935 片、修改 1,829 片）；该结果已被真实裂缝版覆盖。

> 拼接已完成：新层 [TopV, Top) 与现有 [Top, Base+50] 时间区间不相交，
> 直接几何并集。输出 `taigu_supp_merged_v1/merged_dfn_raw_time.vtk`
> （462,976 单元）与 `merged_dfn_patches.csv`（418,024 行，三层标签齐全），
> 可直接用于 Step9 剖面。

> Step9/9b 已完成：
> - 剖面全量 20 图（`sections/overview`、`sections/local_200m`）已生成，纵轴 2354–4232 ms，
>   可覆盖 TopV–Base；原 Step9 配色表只认"上部复合层/太古界风化壳"两层，因此又用
>   "新层临时并入上部复合层"的 CSV 出了 2 张快速验证图（`sections_quick/`），
>   确保新层几何可见；若要在图例里单独显示"上部补充层"，需要新增一个 Step9 配色脚本。
> - Step9b 出三层 300 m 网格面积密度对比图（统一色标）：
>   `taigu_supp_step9b_v1/layer_area_density_compare.png` + `layer_area_density_grid_300m.csv`。

> `vert1` 近直立大尺度口径（2026-11-15）：
> Step7C 对推测大尺度片施加真倾角 `>=65°`、目标约 `75°`、XZ/YZ 视倾角
> `>=30°` 的约束；原始断层几何不变。Step8 使用 `500 m / 100 ms` 的
> 常规测井点位软支持，仅调整置信度并输出
> `large_conventional_control_support.csv`，不移动或旋转大尺度片。
> 补充层合并与多尺度剖面的 `vert1` 产物分别为
> `taigu_supp_merged_vert1/`、`taigu_supp_step9_scale_wiggle_vert1/`。

```bash
python3 太古界/supp_layer/merge_supp_dfn.py \
  --full-csv 太古界/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1_vert1/well_corrected_dfn_fracture_patches.csv \
  --full-vtk 太古界/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1_vert1/well_corrected_dfn_raw_time.vtk \
  --output-dir 太古界/supp_layer/output/taigu_supp_merged_vert1

python3 太古界/supp_layer/plot_supp_step9_scale_wiggle.py \
  --config 太古界/supp_layer/configs/taigu_supp_step9_scale_wiggle.json \
  --dfn-patch-csv 太古界/supp_layer/output/taigu_supp_merged_vert1/merged_dfn_patches.csv \
  --output-dir 太古界/supp_layer/output/taigu_supp_step9_scale_wiggle_vert1
```

> `seg1` 连续分块口径（2026-11-15）：
> 482 个推测大尺度父片沿走向分成 1,205 个相邻子块，每块共享横截面边界。
> 目标块长约 200 m，相邻块最大倾向差 30°、最大倾角差 10°；
> 最终真倾角仍均不低于 65°，Step8 后共享边界最大缝隙为 0 m。
> 输出目录为 `taigu_supp_step9_scale_wiggle_seg1/`。

```bash
python3 太古界/step7c_large_fault_dfn/build_large_fault_dfn.py \
  --config 太古界/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_quanqu1_seg1.json
python3 太古界/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py \
  --config 太古界/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_quanqu1_seg1.json
python3 太古界/step8_well_correction/correct_taigu_multiscale_dfn_with_well_controls.py \
  --config 太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_quanqu1_seg1.json
```

这些步骤同样只新增配置/脚本，不改原代码。
