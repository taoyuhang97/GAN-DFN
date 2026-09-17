# 太古界 Step9 正式多背景剖面

正式入口为 `build_taigu_multibackground_sections.py`，流程结构继承砂砾岩正式
Step9，但使用太古界自己的三属性主道头、OBN独立道头、Top/Mid/Base层位合同
及Step8最终多尺度DFN。

每个范围生成5种背景乘XZ/YZ共10张图：蚂蚁体、相干体、曲率绝对值、OBN
振幅变密度、OBN波形加变面积。`overview`和`local_200m`合计20张。

OBN振幅仅用于Step9展示，不参与Step5-Step8预测。蚂蚁体`-1`按有效弱响应
显示；相干体使用原值且低值显示为深色；曲率正式图使用绝对值。

## 图例方案（2026-09-17 对齐砂砾岩）

图例样式对齐砂砾岩正式主线 `优化阶段二/正式主线/step9_section_visualize`：

- **图例放在数据区外侧**（`fig.legend(..., loc="upper left", bbox_to_anchor=(1.005, 1.0))`），
  不再压在剖面右上角遮挡数据；
- DFN 裂缝片按层段着色：上部复合层 `#ff7a00`（橙）、太古界风化壳 `#00a6ff`（蓝），
  图例文字为 `DFN裂缝片：<层位>`；
- 尺度用**灰阶三档**表示：小尺度 `#9ca3af` / 中尺度 `#6b7280` / 大尺度 `#374151`，
  文字为 `小尺度裂缝 / 中尺度裂缝 / 大尺度裂缝`；
- 成像解释带 `Step3` 前缀：`Step3成像测井裂缝点`（品红三角 `#ff2bd6`）；
- 层位线带代号：`上部复合层顶(T-a-1)`、`太古界顶(Art_1)`、`风化壳底(Art_d1-1)`；
- 井轨迹：`埕北古斜405井轨迹（成像段）`。

运行前接口检查：

```bash
python 太古界/step9_sections/build_taigu_multibackground_sections.py \
  --config 太古界/step9_sections/configs/taigu_step9_multibackground_v2.json \
  --validate-only
```

正式长任务应在tmux中运行，并将标准输出和错误输出写入Step9输出目录的日志。
