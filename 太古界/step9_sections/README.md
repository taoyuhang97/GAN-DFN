# 太古界 Step9 正式多背景剖面

## 2026-09-22 两项修改（md1 轮次）

### (a) 常规测井段 = 目标地层内的井轨迹（需求方口径）

- 旧口径：绿色虚线"埕北古斜405常规测井段"画的是**整条采样轨迹**（含成像解释段，且在
  MD 3998–4180 的未采样空档上做直线插值）。
- 新口径：**只取 `InHorizonLayer==1` 的目标层内井轨迹**，按实际采样点连续绘制，不跨缺口插值；
  目标层之外的区域不再计入常规测井段（成像段由青色"成像测井段轨迹"单独表示）。
- 每张图的 `overlay.conventional_log_interval_ms` 与 `overlay.window_audit` 同步给出
  `conventional_log_time_ms / sample_count / step_median_ms / gap_count / gap_threshold_ms` 与
  `track_time_ms`（完整轨迹）。405 md1 实测：常规测井段 TIME **2950.96–3132.29 ms**、
  7,369 个采样点、**gap_count = 0**（层内数据完整，两期测井互补：2016-06-17 覆盖 4180–4471 m、
  2016-06-10 覆盖 4183–4813 m，4471–4480 m 由 06-10 补齐）。
- 完整井轨迹（黑色）仍按物理路径绘制，跨越未采样空档（`track_time_ms` 2746.84–3132.29 m）；
  它是井斜曲线的积分结果，不代表该段有常规测井采样。

### (b) 图例字体改用简体中文（SC）字面

- 旧实现 `configure_fonts()` 用 `NotoSansCJK-Regular.ttc` + `addfont()`，而 matplotlib 只注册
  ttc 的**第一个字面（JP）**，导致图例里的"复""壳"等字按**日文字形**渲染（结构不同）。
- 新实现 `register_simplified_chinese_font()`：用 fontTools 从 ttc 里挑出名字含 `SC` 的字面
  （Noto Sans CJK ttc 的 face 顺序为 JP/KR/**SC**/TC/HK/Mono…），抽出成
  `~/.cache/taigu_step9_fonts/*.otf` 后注册，`font.sans-serif` 首选该字体；
  取不到 SC 字面时回退旧逻辑并打印告警。运行日志会打印
  `[Step9] 中文字体：Noto Sans CJK SC（简体字面…）`，`section_summary.json` 的
  `contracts.matplotlib_chinese_font` 记录实际使用的字体名。

正式入口为 `build_taigu_multibackground_sections.py`，流程结构继承砂砾岩正式
Step9，但使用太古界自己的三属性主道头、OBN独立道头、Top/Mid/Base层位合同
及Step8最终多尺度DFN。

每个范围生成5种背景乘XZ/YZ共10张图：蚂蚁体、相干体、曲率绝对值、OBN
振幅变密度、OBN波形加变面积。`overview`和`local_200m`合计20张。

OBN振幅仅用于Step9展示，不参与Step5-Step8预测。蚂蚁体`-1`按有效弱响应
显示；相干体使用原值且低值显示为深色；曲率正式图使用绝对值。

## 图例方案（2026-09-21，v7 分组 + v8 产状口径）

图例样式对齐砂砾岩正式主线 `优化阶段二/正式主线/step9_section_visualize`：

- **图例放在数据区外侧**（`fig.legend(..., loc="upper left", bbox_to_anchor=(1.005, 1.0))`），
  不压在剖面上遮挡数据；
- v7 起按**用途分组**（`LEGEND_GROUPS` + `grouped_legend()`），组标题为
  `— 地层界面 — / — 井与测井段 — / — DFN 裂缝 — / — 测井证据 — / — 构造 —`；
- 地层界面线带代号：`上部复合层顶(T-a-1)`、`太古界顶(Art_1)`、`风化壳底(Art_d1-1)`；
- 井与测井段：`埕北古斜405常规测井段`（绿色加粗虚线，Step2 采样区间）、
  `埕北古斜405井轨迹`（黑）；
- DFN 裂缝片按层段着色：上部复合层 `#ff7a00`（橙）、太古界风化壳 `#00a6ff`（蓝），
  文字为 `DFN裂缝片：<层位>`；尺度用**灰阶三档**：小尺度 `#9ca3af` / 中尺度 `#6b7280` /
  大尺度 `#374151`，大尺度额外描边以拉开与小/中尺度的粗细差距；
- 测井证据：`Step3成像测井裂缝点`、`Step3真实成像裂缝解释片（按产状）`、
  `Step3成像裂缝产状（视倾角方向，长度∝倾角）`、`Step4常规测井裂缝点（预测）`；
- 构造单独成组：`原始断层（黄色，非预测）`（非预测层，与预测大尺度裂缝分开）；
- v7 起去掉剖面左缘的"成像测井段 / 常规测井段 / 目标地层内"三行旋转标注（互相遮挡），
  三段区间数值改为只写在 `section_summary.json` 的窗口审计里。

## 成像产状绘制口径（2026-09-21，v8）

剖面图上的产状短线/解释片，方向取**裂缝面与剖面面的交线**，即几何视倾角方向，
由 `apparent_dip_trace()` 统一给出：

- 裂缝面法向 `n = (sinα·sinδ, cosα·sinδ, cosδ)`（度量空间 x=东、y=北、z=上），
  剖面法向 XZ 为 `ŷ`、YZ 为 `x̂`，交线 `t = n × 剖面法向`，统一朝下；
- 与闭式公式 `tanβ=|sinα|·tanδ`（XZ）/ `|cosα|·tanδ`（YZ）逐点一致；
- 旧实现取"倾向矢量去掉视线分量"，向量投影后不在裂缝面内，会把高角度缝压成近水平线
  （405 实测：XZ 中位 21.6°→51.6°、YZ 8.4°→73.6°）；
- 只改绘制，不影响 Step3–Step8 的任何数值，改后只需重跑 Step9。

### 两套方位约定的区别（2026-09-21，v9）

叠加上有两类裂缝片，方位列的含义**不同**，转换写在各函数里：

| 图层 | 列 | 含义 | 处理 |
| --- | --- | --- | --- |
| Step3 成像产状/解释片 | `FracAzimuth` | **真倾向方位**（甲方 LAS：`True Dip Azimuth`，0–360） | 直接交给 `apparent_dip_trace` |
| Step7/Step8 DFN 裂缝片 | `AzimuthDeg` | **走向**（0–180，axial；VTK 几何同样按走向建面） | 先 `+90°` 转倾向，再求交线 |

v9 修正了 DFN 片迹线：旧实现 `vertical = tan(dip)·max(|lateral|,0.08)`，归一化后斜率恒等于
真倾角、且走向接近垂直剖面时被直接拉成竖直（本该接近水平的片子画成竖线）。
改为统一走 `apparent_dip_trace` 后，局部 200 m 剖面里屏角 >60° 的 DFN 段从 7 条降到 0 条。

## 纵向范围口径（2026-09-21，v7 = 方案 B）

v6 及以前：`overview` 与 `local_200m` **共用**一套纵向刻度，取全工区层位合同的
`min(TopTimeMs)` ~ `max(BaseTimeMs)` ± `time_padding_ms`（=20 ms），实测 2604–4018 ms。
后果是局部剖面上 3/4 的纵向是无关区域（层位带只占图高 24–26%），纵向被压 6.9 倍。

v7 起改为**逐剖面自适应**（对齐砂砾岩 `well_curved_section_common.py` 的做法）：

* 用**该剖面自己画出来的三条层位曲线**（`TopTimeMs` / `MidTimeMs` / `BaseTimeMs`）的
  min/max ± `scope_time_padding_ms` 作为纵向范围；
* `overview` 与 `local_200m` **各自算**，不再共用；
* 方案 B 取 `scope_time_padding_ms = 100.0`（层位上下各留 100 ms 上下文）；
* `local_200m` 仍取 `overview` 采样网格的子集（两者都对齐 `section_sample_interval_ms`），
  所以属性/OBN 只采样一次，局部只是行列裁剪，NPZ 缓存与 `section_summary.json` 里
  `time_range_ms_by_scope` 记录各自的范围。

实测（埕北古斜405）：

| | v6（全局共用 2604–4018） | v7（逐剖面自适应 ±100 ms） |
| --- | --- | --- |
| `overview` 纵向范围 | 2604–4018 ms（1414 ms） | 2606–3990 ms（1384 ms） |
| `local_200m` 纵向范围 | 2604–4018 ms（1414 ms） | **2750–3322 ms（572 ms）** |
| 局部图层位带占图高 | 26% / 24%（XZ/YZ） | **65% / 58%** |
| 局部图纵向压缩 | **6.86 倍** | **约 2.7 倍** |
| 全局图纵向压缩 | 0.98 倍 | **约 1.0 倍（近似等比例）** |

对应配置：`configs/taigu_step9_multibackground_v2_v7.json`（v6 配置与产物保留不覆盖）。

## 叠加层顺序与 DFN 分层（2026-09-21，v8）

v7 及以前，DFN 裂缝片的 zorder = **7**，被层位线、常规测井段、成像解释片、成像产状短线、
井轨迹（12.4/12.45）全盖住 —— 井上只看得到真实成像裂缝，看不出"构造裂缝 vs 真实裂缝"的差异。
v8 对齐砂砾岩 `dfn_section_overlay.py` / `build_cheye1_dfn_coherence_sections.py` 的做法：

* **DFN 拆成两层**（`patch_segments` 现在多返回一个 `crossed` 标记）：
  * *剖面真实交线*：按片的走向/倾角算出它在剖面法向（XZ→Y，YZ→X）上的半宽，若片中心到该剖面
    的距离小于半宽，就是这个剖面真正切到的片 → 大尺度描边 zorder **12.9**、交线 **13.1**（最上层）；
  * *井周投影片*：±`*_dfn_projection_half_width_m` 内但剖面没切到的片 → 半透明（alpha 0.42）、
    线宽 ×0.85、zorder **12.7**，图例新增"DFN裂缝片：井周投影（半透明）"。
* **补画青色"成像测井段轨迹"**（`IMAGING_TRACK_COLOR`，zorder 12.5/12.55，压在井轨迹之上），
  对应砂砾岩图上的"Step3成像测井井段轨迹"；图例归入"井与测井段"组。

完整层级（v8）：

| zorder | 图层 |
| --- | --- |
| 6.2 / 6.3 | 原始断层（描边 / 黄线） |
| 8 | 层位线（上部复合层顶 / 太古界顶 / 风化壳底） |
| 10.05 | 常规测井段（绿虚线） |
| 10.4 / 10.5 | 成像解释片（描边 / 片） |
| 11 / 11.1 / 11.2 | 成像裂缝点（三角）／成像产状短线（描边 / 线） |
| 12.4 / 12.45 | 井轨迹（描边 / 黑线） |
| 12.5 / 12.55 | **成像测井段轨迹（青光晕 / 青线）** |
| 12.6 | Step4 常规测井裂缝点（预测） |
| 12.7 | DFN 井周投影片（半透明） |
| 12.9 / 13.1 | **DFN 真实交线（大尺度描边 / 线）** |

实测（v8，XZ/YZ）：局部 200 m 剖面画 3036 / 3102 条 DFN 段，其中真实交线 **527 / 336** 条
（约 17% / 11%），其余为井周投影。summary 的每张图 `overlay` 里新增
`dfn_crossed_count` / `dfn_projected_count` 便于核对。

对应配置：`configs/taigu_step9_multibackground_v2_v8.json`（v7 配置与产物保留不覆盖）；
链条脚本 `run_v8_from_7.sh`。

运行前接口检查：

```bash
python 太古界/step9_sections/build_taigu_multibackground_sections.py \
  --config 太古界/step9_sections/configs/taigu_step9_multibackground_v2.json \
  --validate-only
```

正式长任务应在tmux中运行，并将标准输出和错误输出写入Step9输出目录的日志。


---

## 口径变更（2026-09-22，azi1）

产状统一为 **真倾向方位 0–360（自北顺时针）+ 倾角**，唯一实现见
`太古界/common/orientation_frame/convention.py`（换算规则与验证见该目录 README）。

- `DipAzimuthDeg` = 唯一真值字段；`AzimuthDeg` 降级为派生走向 `(DipAzimuthDeg-90)%180`。
- 第三维为 `TIME`（向下为正）的帧一律走 `*_depth` 变体换算。
- 旧口径开关（`family_azimuth_semantics` / `ridge_azimuth_semantics` /
  `vertex_convention` / `dfn_azimuth_semantics`）已删除，不再保留双口径分支。
- 背景：`太古界/太古界流程梳理与问题记录_20260915.md` §0.37；
  施工方案：`太古界/太古界产状口径统一施工方案_20260922.md`。
