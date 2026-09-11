# 太古界属性版 Demo 网格合同

`build_attribute_demo_grid.py` 在确认的5 km x 5 km Demo范围内生成12.5 m目标网格，
再按X/Y将每个网格点唯一映射到属性主道头。构建过程不读取OBN道头。

正式输出：

- `attribute_demo_grid.csv`：Step6A/Step7A使用的 `TraceIdx,X,Y,IX,IY`。
- `attribute_demo_grid_mapping.npz`：二进制的道序映射。
- `attribute_demo_grid_summary.json`：匹配距离、唯一性和矩形完整性QC。

Step6A还会在运行前重新核对 `TraceIdx`对应的属性道头X/Y，并检查层位合同是否
覆盖全部Demo道。OBN道号不允许进入该合同。
