# Step 1 TaiGuJie Strata Contracts

入口：`build_taigu_strata_contracts.py`，md1 配置
`configs/taigu_step1_contracts_md1.json`，输出
`output/taigu_step1_contracts_md1/`（v1 产物保留）。

## 两类合同

| 文件 | 内容 |
|---|---|
| `regular_well_time_strata_contract.csv` / `regular_time_depth_strata_points.csv` / `regular_time_depth_manifest.csv` | 常规井：本井时深 → TIME↔MD 分段线性；三界面（上部复合层顶 / 太古界顶 / 风化壳底）在井口 X/Y 处的地震 TIME，用于逐行层位分类 |
| `imaging_md_strata_contract.csv` | 成像井：甲方 FMI/XRMI 解释井段（**MD**），按井给出 1–2 段区间 + 分界（如埕北313=4783 m），并声明层属 |
| `imaging_md_strata_qc.csv` / `imaging_md_strata_source_check.csv` | 区间与边界 QC；后者把合同区间端点拿到甲方文件深度轴上找最近值，报告残差 |

## 深度口径（2026-09-22，md1 修正）

- 甲方成像解释深度是**测深 MD**：LAS 参数块 `TLFamily_TDEP = Measured Depth`、
  裂缝参数 DLIS index `BOREHOLE-DEPTH` 且深度道名就是 `MD`、成果图图头"深度（测深）"、
  解释报告"处理井段为 3676.7-4000.0m，**共计 323.3m**"（= 4000.0−3676.7，若为 TVD
  对应井段长度应是 ~477 m）。
- 因此合同字段为 `InterpretedMDMin / InterpretedMDMax / InterpretedMDIntervals / BoundaryMD`，
  配置键为 `imaging_md_contracts` / `interpreted_md_intervals` / `boundary_md`。
  旧键名（`imaging_tvd_contracts` 等）仍可读，用于复现历史版本。
- 数值与 v1 **逐字一致**——v1 数值本来就是甲方测深，只是被写成了 TVD 字段。

## 来源核对 QC

`source_depth_checks` 对每口成像井指定甲方源文件与深度列（405=`MD`、
313/169=`TDEP`、310/816/古7/102=空白分隔的首列），把合同区间端点拿到该文件的
深度轴上找最近值：

| 状态 | 含义 |
|---|---|
| `pass` | 两端残差 ≤ `tolerance_m`（默认 1 m）：合同数值就取自这条深度轴 |
| `pass_report_interval` | 两端残差 ≤ `report_tolerance_m`（默认 30 m）：合同取自报告井段，与文件轴端点略有出入 |
| `diff_exceeds_tolerance` | 真正的深度轴错位（把 MD 数值当 TVD 用会差数百米） |

md1 实测：8 行源核对全部 `pass`，最大端点残差 0.10 m（顶）/ 0.25 m（底）。

## 运行

```bash
python3 太古界/step1_strata_contracts/build_taigu_strata_contracts.py \
  --config 太古界/step1_strata_contracts/configs/taigu_step1_contracts_md1.json --replace-output
```
