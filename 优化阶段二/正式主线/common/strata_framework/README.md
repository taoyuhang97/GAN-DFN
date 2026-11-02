# Strata Framework（地层框架）

把"有哪些层位面、哪些地层单元、每层用什么编码/颜色"从代码里挪到配置里，
让各步骤只遍历 `framework.layers`，不再出现 `if layer == "沙三段"` 这类判断。
后续新增地层时，主要改配置而不是改算法。

## 1. 三个概念

| 概念 | 含义 | 例子 |
|---|---|---|
| `Surface` | 一个层位界面 | 砂砾岩 `T4/T5/T6/T7`；太古界 `Top/Mid/Base` |
| `Layer` | 由上下两个界面圈出的地层单元 | 沙三段（T4→T6）、太古界风化壳（Mid→Base） |
| `StrataFramework` | 某工区的一整套定义 | `shaligyan_v1`、`taigu_v1` |

`LayerCode` 统一为 **int**，在配置里显式声明，同一工区内唯一。

## 2. 现有框架

| id | 工区 | 层位面 | 地层（code） |
|---|---|---|---|
| `shaligyan_v1` | 砂砾岩 | T4 / T5 / T6 / T7 | 沙三段(3)、沙四段(4) |
| `taigu_v1` | 太古界 | Top / Mid / Base | 上部复合层(1)、太古界风化壳(2) |

太古界框架通过 `legacy_column` 兼容历史 npy：

| 框架 Surface | 历史列名 |
|---|---|
| `Top` | `T4` |
| `Mid` | `T6` |
| `Base` | `T7` |

（历史 npy 里的 `T5` 是 `T6` 的重复槽位，框架不再表达它。）

## 3. 用法

```python
from common.strata_framework import load_framework, resolve_framework

fw = load_framework("shaligyan_v1")

fw.layer_names()        # ('沙三段', '沙四段')
fw.layer_codes()        # (3, 4)   —— 全部是 int
fw.layer_code_map()     # {'沙三段': 3, '沙四段': 4}

layer = fw.layer("沙三段")        # 也可 fw.layer(3) / fw.layer("Shasan")
layer.top, layer.base             # ('T4', 'T6')
layer.interval_label()            # 'T4->T6'
layer.inclusive_base              # False（层底不含）

# 从步骤配置解析（配置里可写 strata_framework / strata_framework_id）
fw = resolve_framework(config)
```

`legacy_layer_aliases` 让历史调用继续可用：

```python
fw.layer("Shasan")   # == fw.layer("沙三段")
fw.layer("Shasi")    # == fw.layer("沙四段")
```

## 4. 新增一个地层

1. 在 `configs/<area>.json` 的 `surfaces` 里补齐上/下界面；
2. 在 `layers` 里加一条，指定 **int** `code`、`name`、`top`、`base`；
3. 重新构建层位契约（`build_horizon_contract_v2.py`）与层窗（2ms/10ms）；
4. Step2–5 按 `LayerGroup` 自动带上新层；
5. Step6 需要为新层重新训练两阶段模型（这是唯一无法自动化的部分）；
6. Step7–9 按 `LayerGroup` / `LayerCode` 自动分组，配色与图例来自 `color`。

## 5. 当前迁移状态

已完成（第 1 阶段）：

- 新增本模块与两份框架配置；
- `common/horizon_trace_table/horizon_contract.py` 改为框架驱动，
  公开 API 与旧字段（`T4..T7`、`ShasanPresent/ShasiPresent`）保持兼容；
- 5 处 `LAYER_CODE` 浮点定义（`3.0/4.0`）统一为框架 int，并去掉重复声明；
- Step5B/Step6/Step7B-initial/Step8 的 `ALLOWED_LAYERS` 改为框架派生；
- 等价性验证脚本：`scripts/verify_strata_framework.py`。

待做（后续阶段）：

- Step1 `surface_tools.py` 的 `SURFACE_CODES`、Step2 的 `T4Time..T7Time` 列；
- Step3/4 的地层分组与专家库注册；
- Step6B/C 的 `T4-T7 / Shasan / Shasi` 掩膜分支；
- Step7A 的 2 处层段映射、Step9 的 `T4->T6` 图例字符串；
- Step9b 的 `LAYERS / LAYER_HORIZONS`；
- 太古界 `build_taigu_horizon_npy.py` 的 npy schema（改为 Top/Mid/Base）。

## 6. 验证

```bash
python3 scripts/verify_strata_framework.py --traces 3000
```

检查项：LayerCode 全为 int、层引用三种写法解析一致、框架层窗掩膜与旧硬编码逐元素相等、
`surface_grids_from_contract` 键集合不变、太古界 legacy 列映射正确、代码中无浮点 LayerCode。
