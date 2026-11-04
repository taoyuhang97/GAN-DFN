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
    └── taigu_supp_step7_v1/
        ├── supp_demo_grid_block.csv            # 401×401 区块网格（局部 IX/IY）
        ├── supp_window_code.npz                # 新层层窗（160801×494）
        ├── supp_7a_config.json ... supp_7d_config.json
        ├── step7a_supp/                        # 小尺度 43,061 片
        ├── step7b_supp/                        # 中尺度 3,919 片
        ├── step7c_supp/                        # 大尺度 159 片
        └── step7d_supp/                        # 融合 47,139 片 + 修正层标签 CSV
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

## 7. 与原代码的接口

- 新框架通过 `supp_framework.load_supp_framework()` 注册后，
  `common/horizon_trace_table/horizon_contract.py` 即可按 `taigu_supp_v1`
  读取 TopV/Top/Mid/Base，**无需改代码**；
- `build_supp_horizon.py` 直接调用 `load_framework_file()`，不依赖注册表；
- 现有 `demo_horizon_contract.npy`、现有 DFN、现有 Step9/9b 产物**均未改动**。

## 8. 后续步骤（尚未实现）

1. **Step8**：用 405 在新层内的成像点做井控校正；
2. **拼接**：把新层 DFN（区块）与现有全域 DFN 做几何并集，交界处按 `Top − ε` / `Top` 切齐；
3. **Step9 / Step9b**：新增"上部补充层"剖面与 P33 图，和现有两层并排对比。

> Step6A 已完成：`predict_supp_density.py` 复用"上部复合层"两阶段模型，
> 属性分数使用其 normalization contract；层窗取 `[TopV, Top)`，与上部复合层不重叠。

> Step6B/C/D 已完成：`build_supp_legacy_adapter.py` 把新层映射到 T4/T6/T7 槽位
> （T4=TopV、T6=Top−1 ms、T7=Top），原有 6B/6C/6D 脚本无需修改即可运行；
> 输出的 LayerGroup 仍是临时"沙三段"语义，Step7 之后会统一改回"上部补充层"。

> Step7A–D 已完成：7A/7B/7C 复用原有脚本（新区块配置）；7D status=pass，
> 融合 47,139 片。`relabel_supp_dfn.py` 已把层标签统一为"上部补充层"/LayerCode=3。

这些步骤同样只新增配置/脚本，不改原代码。
