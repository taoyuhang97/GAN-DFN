# Step4 常规测井裂缝预测当前正式文档

更新时间：2026-07-28

状态：**本文件是 Step4 唯一有效的设计、结果和使用说明。**

Step4 当前正式版本为 `formal_six_expert_library_v3`。其他模型目录、旧脚本和历史结果只用于回顾或对照，不得作为 Step5 及后续步骤的数据入口。

## 1. 当前任务与边界

Step4使用Step3成像测井裂缝标签训练模型，再对Step2真实井T4-T7区间中具备完整同测次GR和电阻率曲线的深度行进行裂缝预测。

当前固定使用三组输入：

| PairType | 统一输入 | 参考探测角色 | 直接成像监督 |
|---|---|---|---|
| `LLD_LLS` | GR + LLD + LLS | 深侧向 + 浅侧向 | 车页1导眼 |
| `RD_RS` | GR + RD + RS | 深电阻率 + 浅电阻率 | 车151HF、车662、车663 |
| `RILD_RILM` | GR + RILD + RILM | 深感应 + 中感应 | 当前没有 |

训练和预测均按井内 `DEPT` 精确连接：

```text
Step3 formal_rebuild成像标签
+
Step2 *_t4_t7_real_well_gr_resistivity.csv
```

没有完整三曲线、超出曲线有效区间或空间坐标无效的行不预测、不填零，也不进入正式合并曲线，只在QC中记录。

## 2. 三组电阻率之间的关系

三组曲线不能直接改名后混入同一个无差别模型。

可以统一的是：

- GR、深探测电阻率和参考电阻率的物理角色。
- 各曲线井内地层稳健Z值、分位秩、局部异常、梯度和多尺度残差。
- GR与电阻率背景或局部异常之间的交互。

不能直接统一的是：

- `LLD-LLS`、`RD-RS`和`RILD-RILM`的固定数值差。
- 深浅比与深中比。
- 差值的绝对大小和正负方向。
- 一个固定的LLD到RD、LLD到RILD换算公式。

### 2.1 RD/RS与RILD/RILM桥接

车55和车63提供了目前可靠的RD/RILD共同区间：

| 井 | 约1m共同深度箱 | 深曲线相关 | 参考曲线相关 | 说明 |
|---|---:|---:|---:|---|
| 车55 | 781 | 0.973 | 0.969 | 局部异常和去趋势后仍保持强相关 |
| 车63 | 404 | 0.966 | 0.967 | 同一LAS，GR零偏移一致 |

跨井稳健映射的验证R2约为：

- RD到RILD：0.636-0.840。
- RS到RILM：0.787-0.792。

因此RILD/RILM可以学习RD/RS专家的教师信号，但这只能证明测量域之间可桥接，不能替代真实成像裂缝标签。

### 2.2 LLD/LLS桥接限制

车57、车74、车15-2和车254的共同区间较短，局部异常关系弱或方向不稳定。车斜576属于错深或曲线别名复制，不能作为跨仪器标定证据。

当前结论是：LLD/LLS必须保留独立专家，并依靠车页1导眼直接标签训练。它目前只有一口标签井，尚未完成真正的跨井验证。

### 2.3 数据风险

RD/RS数据中存在负值和特殊大负值。当前处理原则为：

- Step2保留原始值，不恢复高值或饱和值删除规则。
- 非正电阻率不直接计算物理对数。
- Step4使用signed-log、稳健缩放、分位秩和非正值标志。
- 不允许因非正值而静默删除整段连续测井数据。

## 3. 当前六专家模型

模型库固定包含6个专家：

| 地层 | LLD/LLS | RD/RS | RILD/RILM |
|---|---|---|---|
| 沙三段 | `沙三段_LLD_LLS` | `沙三段_RD_RS` | `沙三段_RILD_RILM` |
| 沙四段 | `沙四段_LLD_LLS` | `沙四段_RD_RS` | `沙四段_RILD_RILM` |

每个专家分别训练模型、阈值和密度校准，不共享最终XGBoost参数。

两阶段模型为：

```text
Stage 1: XGBClassifier -> PredFractureProb
Stage 2: XGBRegressor  -> PredConditionalDensity
PredDensity = PredFractureProb * PredConditionalDensity
```

### 3.1 标签

Stage 1使用三类标签：

- 明确裂缝：高于本专家正密度中位数、真实裂缝点所在行，或真实点前后0.5m。
- 明确背景：`Density=0`且距离真实裂缝点超过0.75m。
- 不确定区：低密度插值和裂缝边缘，不参与Stage 1分类。

Stage 2继续使用全部 `Density>0` 行拟合条件密度。

| 专家 | 明确裂缝 | 明确背景 | 不确定 |
|---|---:|---:|---:|
| 沙三LLD/LLS | 495 | 1,111 | 266 |
| 沙四LLD/LLS | 38 | 74 | 16 |
| 沙三RD/RS | 712 | 7,135 | 197 |
| 沙四RD/RS | 141 | 807 | 49 |

### 3.2 特征

当前特征包括：

- GR、深探测和参考电阻率的稳健Z值和井内地层分位秩。
- 电阻率signed-log及正值物理log。
- 0.5、1、3、5m局部中位数残差。
- 1、3、5m局部MAD。
- 深度梯度。
- 深浅或深中差、归一化差异及局部异常。
- GR与电阻率异常交互。
- 非正电阻率标志。

### 3.3 当前验证结果

| 专家 | 验证方式 | ROC AUC | Average Precision |
|---|---|---:|---:|
| 沙三LLD/LLS | 单井连续深度五折 | 0.5413 | 0.3797 |
| 沙四LLD/LLS | 单井连续深度五折 | 0.5119 | 0.6644 |
| 沙三RD/RS | 留一井 | 0.5280 | 0.1035 |
| 沙四RD/RS | 留一井 | 0.6782 | 0.3105 |
| 沙三RILD/RILM | 留一桥接井，对RD教师 | 0.7057 | 0.0317 |
| 沙四RILD/RILM | 留一桥接井，对RD教师 | 0.9283 | 0.7147 |

能力边界：

- 沙三RD/RS跨井判别能力较弱。
- LLD/LLS只有一口直接标签井，现有指标不是跨井泛化结果。
- RILD/RILM指标只衡量对RD教师的复现，不是对真实成像裂缝的验证。
- 当前结果是已完成数据合同和发布验收的实验版本，不能表述为高精度裂缝预测模型。

## 4. 正式预测与确定性合并

每个支持的PairType先独立预测。专家明细唯一键为：

```text
WellName + DEPT + InputPairType
```

最终合并曲线唯一键为：

```text
WellName + DEPT
```

同一井深存在多个专家时固定选择：

```text
RD_RS > LLD_LLS > RILD_RILM
```

当前不平均概率或密度。专家切换、地层变化或深度缺口超过0.5m时重新切分 `SupportSegmentID`。裂缝点只从合并曲线重新提取，不能拼接各专家裂缝点。

### 4.1 正式结果规模

结果目录：

```text
output/formal_six_expert_library_v3
```

当前正式发布：

- 6个专家，12个模型文件。
- 38口预测井。
- 259,788条专家明细。
- 250,368条唯一井深合并预测。
- 9,420个井深位置存在多专家覆盖。
- 4,114个合并后离散裂缝点。
- 23项自动发布验收全部通过。

| 曲线组 | 专家明细行 | 最终被选行 |
|---|---:|---:|
| LLD/LLS | 62,832 | 62,563 |
| RD/RS | 148,117 | 148,117 |
| RILD/RILM | 48,839 | 39,688 |

逐井目录结构：

```text
predictions/real_well_predictions/<WellName>/
├── <WellName>_LLD_LLS_expert_prediction.csv   # 有该专家时
├── <WellName>_RD_RS_expert_prediction.csv     # 有该专家时
├── <WellName>_RILD_RILM_expert_prediction.csv # 有该专家时
└── <WellName>_merged_density_prediction.csv
```

## 5. 裂缝比例检查与当前问题

### 5.1 离散裂缝点比例

| 地层 | 常规测井预测 | 成像测井样本 | 判断 |
|---|---:|---:|---|
| 沙三段 | 3,133 / 220,855 = 1.419% | 160 / 9,916 = 1.614% | 较接近，预测略低 |
| 沙四段 | 981 / 29,513 = 3.324% | 20 / 1,125 = 1.778% | 预测约为成像的1.87倍 |
| 合计 | 4,114 / 250,368 = 1.643% | 180 / 11,041 = 1.630% | 总体接近 |

总体比例接近不能作为独立精度证据。当前 `point_count_scales` 使用成像测井中“真实点数 / 密度积分”按地层标定，再将预测密度转换成离散点；同时每个二值阳性支持段至少产生一个点。因此总体1.643%接近成像1.630%，部分是后处理主动校准的结果。

当前最明显的问题是沙四离散点比例偏高，说明总体比例掩盖了地层差异。

### 5.2 二值裂缝记录比例

| 口径 | 常规测井预测 | 成像测井 |
|---|---:|---:|
| `HasFracture=1 / 总记录` | 22,741 / 250,368 = 9.083% | 明确裂缝 1,386 / 11,041 = 12.553% |
| 明确裂缝 / 可判定样本 | 不适用 | 1,386 / 10,513 = 13.184% |
| 连续插值 `Density>0` | 不能作为二值比例，预测密度几乎均为正 | 1,823 / 11,041 = 16.511% |

按曲线组和地层比较：

| 专家域 | 预测二值比例 | 成像明确裂缝/可判定样本 | 当前判断 |
|---|---:|---:|---|
| LLD/LLS 沙三 | 25.53% | 30.82% | 略低 |
| LLD/LLS 沙四 | 28.81% | 33.93% | 略低，但标签样本很少 |
| RD/RS 沙三 | 1.66% | 9.07% | 明显偏低 |
| RD/RS 沙四 | 24.31% | 14.87% | 明显偏高 |
| RILD/RILM 沙三 | 0.00% | 无直接成像标签 | 无法验证 |
| RILD/RILM 沙四 | 16.03% | 无直接成像标签 | 无法验证 |

因此不能用总体比例判断模型已经吻合。当前主要问题为：

1. 沙三RD/RS二值裂缝比例过低。
2. 沙四RD/RS和沙四离散点比例偏高。
3. 沙三RILD/RILM没有二值阳性。
4. LLD/LLS标签井数量不足。
5. RILD/RILM没有直接成像监督。
6. 总体离散点比例受后处理校准影响，不能作为模型精度验证。
7. 不同井、不同地层的GR和电阻率响应方向并不完全一致。
8. 沙三直接专家留井AUC接近随机水平，可能存在标签、错深、特征方向或跨井域差异问题。

## 6. Step5及后续步骤唯一输入

唯一入口：

```text
output/formal_six_expert_library_v3/six_expert_prediction_manifest.json
```

Step5只能读取manifest中的：

```text
prediction_table =
predictions/all_wells_t4_t7_merged_density_prediction.csv
```

禁止：

- 将 `all_wells_t4_t7_expert_detail_prediction.csv` 作为Step5输入。
- 扫描逐井目录后拼接专家文件。
- 将缺少电阻率覆盖的井段补成零裂缝。
- 将连续 `Density>0` 直接解释为二值裂缝。
- 将RILD教师迁移指标解释成成像测井验证精度。
- 使用v1、v2或旧专家库manifest替代当前v3 manifest。

## 7. 下一轮改进优先级

1. 重新检查车151HF、车662、车663沙三标签与GR/RD/RS的局部错深、标签连续插值范围和响应方向。
2. 分专家、分地层重新校准Stage 1阈值，不再依赖总体比例；重点修正RD/RS沙三偏低和沙四偏高。
3. 重构离散点生成校准，分别检查密度积分转换、每段至少一个点规则和地层采样长度影响。
4. 增加事件级指标、允许深度误差的Precision/Recall、1m深度箱指标和深度分块Bootstrap。
5. 为LLD/LLS增加至少一口直接成像标签井，建立真正的留一井验证。
6. 为RILD/RILM增加直接成像标签；在此之前单独报告教师迁移结果和真实标签结果。
7. 在相同数据切分和标签下对比XGBoost、CatBoost、ExtraTrees及TCN，先判断问题来自模型还是数据。
8. 在可靠多曲线共同区间评估专家一致性和不确定性；验证通过前继续使用当前确定性选择，不做静默平均。
9. 所有改进版本写入新版本目录，通过隔离验收后再替换当前正式结果。

## 8. 运行与验收

正式运行：

```bash
python 优化阶段二/正式主线/step4_expert_real_well_prediction/train_and_predict_six_expert_library.py \
  --config 优化阶段二/正式主线/step4_expert_real_well_prediction/configs/formal_six_expert_library_v3.json \
  --replace-existing-output
```

关键结果：

- `six_expert_prediction_manifest.json`：下游唯一入口。
- `step4_six_expert_acceptance_summary.json`：正式验收结果。
- `model_library/six_expert_registry.csv`：6个专家注册表。
- `validation/direct_expert_validation_metrics.csv`：直接专家验证。
- `validation/rild_expert_bridge_validation_metrics.csv`：RILD教师迁移验证。
- `qc/well_expert_prediction_coverage.csv`：逐井覆盖。
- `qc/unsupported_pair_intervals.csv`：无预测区间。
- `predictions/all_wells_t4_t7_merged_density_prediction.csv`：Step5正式输入。
- `predictions/all_wells_t4_t7_merged_fracture_points.csv`：合并后裂缝点。
- `predictions/all_wells_t4_t7_expert_detail_prediction.csv`：诊断明细，不供Step5使用。

验收必须确认：

- 6个专家均已注册。
- 专家明细唯一键为 `WellName + DEPT + InputPairType`。
- 合并表唯一键为 `WellName + DEPT`。
- 合并选择符合 `RD_RS > LLD_LLS > RILD_RILM`。
- 每行源曲线完整，坐标和密度有限。
- 支持段不跨越大于0.5m的缺口。
- 裂缝点只属于合并支持段。
- 每口井文件集合与实际可用专家一致。
- Step5 manifest只指向合并表。

## 9. 历史版本状态

| 版本 | 状态 | 使用限制 |
|---|---|---|
| 旧井级专家库 | 历史基线 | 不作为当前Step5输入 |
| `formal_gr_resistivity_lateral_v1` | 历史实验 | 保留对照，不作为当前输入 |
| `formal_gr_resistivity_strata_library_v2` | 预测已判定无效 | 只保留模型和训练/验证产物 |
| `formal_six_expert_library_v3` | 当前正式版本 | 只能通过当前manifest使用 |

诊断脚本和历史配置仍可用于回顾，但本README之外不再保留独立Step4说明文档。任何与本文件冲突的历史对话、旧结果说明或旧manifest均以本文件和当前v3正式manifest为准。
