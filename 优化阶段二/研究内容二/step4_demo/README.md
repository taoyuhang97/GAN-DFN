# 第四步最小 Demo

本目录用于实现“第四步：校准属性、断层与裂缝关系”的小范围最小闭环 demo。

当前版本只做一件事：

- 输入一张统一样本表
- 按层位分组
- 对指定属性做最基本的“属性值 -> 裂缝密度”分箱统计
- 输出后续可直接汇报或继续建模的关系结果

这不是完整第四步实现，也不负责全区推广。它的目标是先把下面这条链路跑通：

```text
统一样本表 -> 分层样本筛选 -> 属性关系统计 -> 输出关系表
```

## 1. 输入要求

输入文件建议为 `csv`，一行表示一个样本点。样本可来自真实井或第三步生成的近井虚拟井。

### 必需字段

- `LayerGroup`
  - 分层标签
  - 当前建议值：`sha3` / `sha4`，也兼容其他字符串
- `FractureDensity`
  - 连续裂缝密度值
  - 建议范围 `[0, 1]`

### 推荐字段

- `PointConfidence`
  - 样点级可信度
  - 若缺失，则默认所有样本权重为 `1.0`
- `SourceType`
  - 样本来源
  - 例如 `real` / `virtual`
- `FaultDistance`
  - 到断层距离
  - 可作为一个普通属性参与统计
- 其他属性列
  - 例如 `SEIS_TRUE`
  - `COHERENCE`
  - `ANT_TRACK`
  - `CURVATURE_MAX`
  - `CURVATURE_POS`
  - `FRACTURE_INV`

### 不建议放入属性列表的字段

- 主键类字段，如 `SampleID`
- 标识类字段，如 `WellName`
- 坐标类字段，如 `X/Y/T`
- 纯文本备注字段

## 2. 最小功能

当前 demo 支持：

- 按层位分别统计
- 对每个属性做分箱
- 输出每个分箱内的：
  - 样本数
  - 加权样本数
  - 属性均值
  - 裂缝密度加权均值
  - 裂缝密度加权标准差
- 输出每个属性在每个层位下的摘要：
  - 样本数
  - 有效样本数
  - 加权 Pearson 相关系数
  - 属性均值、标准差
  - 裂缝密度均值、标准差

## 3. 运行方式

### 3.1 自动识别属性列

```bash
python 优化阶段二/研究内容二/step4_demo/run_step4_demo.py \
  --input-csv /path/to/unified_samples.csv \
  --output-dir /path/to/step4_demo_output
```

### 3.2 手动指定属性列

```bash
python 优化阶段二/研究内容二/step4_demo/run_step4_demo.py \
  --input-csv /path/to/unified_samples.csv \
  --output-dir /path/to/step4_demo_output \
  --attribute-cols SEIS_TRUE COHERENCE ANT_TRACK CURVATURE_MAX CURVATURE_POS FRACTURE_INV FaultDistance
```

### 3.3 指定设备

```bash
python 优化阶段二/研究内容二/step4_demo/run_step4_demo.py \
  --input-csv /path/to/unified_samples.csv \
  --output-dir /path/to/step4_demo_output \
  --device cuda
```

说明：

- 若环境中安装了 `torch` 且 `CUDA` 可用，可使用 `--device cuda`
- 若未安装 `torch`，程序会自动退回到 `numpy/pandas` 路径

## 4. 输出产物

运行完成后，输出目录下至少会生成：

- `relation_bins.csv`
  - 分层、分属性、分箱后的关系明细
- `relation_summary.csv`
  - 分层、分属性摘要结果
- `run_config.json`
  - 本次运行的参数记录

## 5. 输入字段默认约定

默认列名如下，可通过 CLI 修改：

- 层位列：`LayerGroup`
- 目标列：`FractureDensity`
- 权重列：`PointConfidence`

## 6. 当前版本的局限

- 只支持 `csv`
- 只做一维属性关系统计，不做多属性联合校准
- 不直接生成全区裂缝发育倾向体
- 不做图件输出
- 分箱默认按分位数切分，主要用于小范围 demo

## 7. 后续可扩展点

后续可以在当前骨架上继续扩展：

1. 增加 Parquet 输入，减轻大样本表读写压力
2. 接入 PyTorch DataLoader，支持更稳定的 GPU 批量统计
3. 从单属性关系扩展到多属性联合评分
4. 输出“分层裂缝发育倾向分数”中间结果
5. 接入第五步的小范围空间化 demo
