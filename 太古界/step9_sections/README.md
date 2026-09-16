# 太古界 Step9 正式多背景剖面

正式入口为 `build_taigu_multibackground_sections.py`，流程结构继承砂砾岩正式
Step9，但使用太古界自己的三属性主道头、OBN独立道头、Top/Mid/Base层位合同
及Step8最终多尺度DFN。

每个范围生成5种背景乘XZ/YZ共10张图：蚂蚁体、相干体、曲率绝对值、OBN
振幅变密度、OBN波形加变面积。`overview`和`local_200m`合计20张。

OBN振幅仅用于Step9展示，不参与Step5-Step8预测。蚂蚁体`-1`按有效弱响应
显示；相干体使用原值且低值显示为深色；曲率正式图使用绝对值。

运行前接口检查：

```bash
python 太古界/step9_sections/build_taigu_multibackground_sections.py \
  --config 太古界/step9_sections/configs/taigu_step9_multibackground_v2.json \
  --validate-only
```

正式长任务应在tmux中运行，并将标准输出和错误输出写入Step9输出目录的日志。
