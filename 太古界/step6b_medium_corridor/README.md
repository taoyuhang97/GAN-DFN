# Step6B 太古界中尺度裂缝带

本目录只负责从曲率体、蚂蚁体和相干体中提取中尺度裂缝带先验。
Step6A密度体只提供属性网格、时间轴和SGY输出模板，不参与中尺度评分。
代码、配置和输出与 Step6C、Step6D 分离。

当前正式配置为 `configs/taigu_step6b_medium_v7_anttrack_led.json`（2026-09-14 定稿）：

- 只接受 Common 属性 Demo 网格中的属性体 `TraceIdx`，禁止OBN道号；
- 启动前逐道核对Step6A输出道序、属性道头X/Y和层位合同；
- 曲率体、蚂蚁体和相干体统一转换到从1800 ms开始的绝对TWT时间轴；
- 蚂蚁体 `-1` 是有效弱/无裂缝响应，不是缺失值；
- 三属性按Common v2合同分复合层和风化壳归一化，不另存完整归一化体。
- 中尺度评分采用蚂蚁体0.70、低相干0.20、曲率0.10；
- 最小组件50体素、最小倾角20度、最小线性1.12、最小平均得分0.55；
- 25至49体素的小碎片只有在蚂蚁体平均得分不低于0.80时才允许桥接恢复。

`output/taigu_medium_v1`~`v6` 及其配置（含 v4/v5 所依赖的三属性合同 v2 分支）
已于 2026-09-15 移入 `太古界/_archive_20260915/`，不再作为 Step7B 输入。
当前只有 `output/taigu_medium_v7_anttrack_led` 是正式结果。

正式运行：

```bash
python 太古界/step6b_medium_corridor/code/build_step6b_medium_corridor_prior.py \
  --config 太古界/step6b_medium_corridor/configs/taigu_step6b_medium_v5_scale_separation.json
```


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
