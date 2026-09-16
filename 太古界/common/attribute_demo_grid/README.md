# 太古界属性版 Demo 网格合同

`build_attribute_demo_grid.py` 在确认的5 km x 5 km Demo范围内生成12.5 m目标网格，
再按X/Y将每个网格点唯一映射到属性主道头。构建过程不读取OBN道头。

正式输出：

- `attribute_demo_grid.csv`：Step6A/Step7A使用的 `TraceIdx,X,Y,IX,IY`。
- `attribute_demo_grid_mapping.npz`：二进制的道序映射。
- `attribute_demo_grid_summary.json`：匹配距离、唯一性和矩形完整性QC。

Step6A还会在运行前重新核对 `TraceIdx`对应的属性道头X/Y，并检查层位合同是否
覆盖全部Demo道。OBN道号不允许进入该合同。

## Demo 范围变更（2026-09-16）：移到 405 居中

原范围中心 (663000, 4240900)，埕北古斜405 落在东北角（距东边仅 76 m、距北边 790 m），
Step9 剖面不便观察。现移动为：

| 项 | 值 |
|---|---|
| 名称 | `taigu_attribute_demo_5km_405center` |
| 中心 | (665425, 4242612.5) |
| 范围 | X [662925, 667925]，Y [4240112.5, 4245112.5]（实际道网 Y 取整为 4240113–4245113） |
| 405 位置 | 距西 2499 m / 东 2501 m / 南 2497 m / 北 2503 m（基本正中） |
| 道数 | 160801（401 × 401，12.5 m） |

构建命令（产物在 `output/` 下，按 `.gitignore` 不入库，需本地重建）：

```bash
python3 太古界/common/attribute_demo_grid/build_attribute_demo_grid.py \
  --attribute-trace-header-csv 太古界/common/attribute_trace_contract/output/taigu_attribute_trace_header_v1/attribute_trace_header.csv \
  --output-dir 太古界/common/attribute_demo_grid/output/taigu_attribute_demo_grid_v2_405center \
  --center-x 665425 --center-y 4242612.5 --half-size-m 2500 --grid-spacing-m 12.5
```

匹配 QC：最大距离 1.414 m（容差 2.0 m）、160801 道一对一、矩形完整。

**代价**：新范围内只有 埕北古斜405（eligible、有 Step4 预测）、埕北古4、埕北805
（后两口 Step2 状态 rejected，无预测）；原范围内的 埕北古403/埕北古406 已出框，
其中 406 是原来区内第二口有预测的井。若需要更多井控证据，可改为只东移（保持 Y 范围），
此时 405 距北边约 790 m 仍可观察。
