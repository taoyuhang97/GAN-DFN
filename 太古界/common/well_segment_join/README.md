# 太古界公共测井段回接模块（well_segment_join）

## 解决的问题（P0-4）

太古界一口井有多个测井段（可能重叠、相邻或存在空档，例如埕北310 两段网格相差 0.042 m、
埕北古斜405 两段重叠 288 m）。Step3 成像组行和 Step4 合并预测点**都不带 X/Y/TIME**，
只能按 MD 回到 Step2 段文件取坐标。

改造前这套逻辑在 5 处各写一遍，容差分别写死为 0.011 m（4 处）和 0.05 m（1 处），
未匹配的点直接 `dropna` / `continue` 静默丢弃。实测后果：Step4 的 4553 个点里有 11 个
直接消失（含埕北310 全部 9 个），另外 405 有 250/385 个点落在 0.005–0.010 m 的
"擦边区"，只差 0.001 m 就会全部丢失。

## 约定

1. **一个输入点产出一行**，行序与输入一致，不重排、不重新编号；
2. 选段优先级：行内显式来源段（如 `InputSegmentPath`）> `preferred_ids_by_row`（如
   Step4 来源明细回溯）> |ΔMD| 最小 > 段序（稳定 tie-break）；
3. 容差默认按该井 **MD 半采样步长**自动推导（0.125 m 步长 → 0.0625 m），
   下限 0.02 m、上限 0.5 m 可配；
4. 未命中点**保留在结果里**，原因写入审计表；
5. 审计字段统一为 `MD_JOIN_AUDIT_COLUMNS`，summary 由 `summarize_join()` 生成。

## 使用

```python
from common.well_segment_join import (
    attach_geometry_by_md, cached_well_segment_pool, summarize_join,
)

matched, audit = attach_geometry_by_md(
    points,                                  # 需含 WellName / MD（可选 TVD、StrataName）
    lambda well: cached_well_segment_pool(samples_root, well),
    preferred_column="InputSegmentPath",     # Step3 成像组用
    geometry_source="step3_imaging_gt",
)
summary_block = summarize_join(audit)
```

Step4 合并点（无显式来源列）额外传 `Step4SourceSegmentLookup`：

```python
lookup = Step4SourceSegmentLookup(step4_source_detail_csv, tvd_tolerance_m=0.25)
preferred = pd.Series(
    [";".join(lookup.preferred_segment_ids(well, tvd)) for well, tvd in zip(points["WellName"], points["TVD"])],
    index=points.index,
)
matched, audit = attach_geometry_by_md(
    points, pool_resolver, preferred_ids_by_row=preferred, geometry_source="step4_predicted",
)
```

## 接入位置

| 步骤 | 用途 |
|---|---|
| Step5 | 成像强监督逐段回接（`build_taigu_virtual_wells.py`） |
| Step7A | 多井方位族取点（`load_multiwell_orientation_points`）与成像密度剖面 QC |
| Step8 | Step4 常规井控制点 + Step3 成像真值点回接 |
| Step9 | 剖面成像点回接 |

## 自检

```bash
python3 太古界/common/well_segment_join/smoke_segment_join.py
```

对埕北310 / 埕北古斜405 / 埕北313 / 埕北古10 构造用例，断言
"输入点数 = 匹配数 + 未命中数、无重复行、行序不变"。
