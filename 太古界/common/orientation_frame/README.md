# orientation_frame：太古界产状口径的唯一实现

**口径（2026-09-22 起，全链唯一标准）**

| 量 | 定义 |
| --- | --- |
| `DipAzimuthDeg` | 真倾向方位，`[0,360)`，自北顺时针（缝"往哪边倒"）。**唯一真值字段** |
| `DipDeg` | 倾角 `(0,90]` |
| `StrikeDeg` / `AzimuthDeg` | 派生走向 `(DipAzimuthDeg-90)%180`，兼容字段，不参与几何计算 |
| 几何帧 | `x=东、y=北、z=上`；`TIME` 向下为正，1 ms = 2 m |

## 为什么需要它

链上曾同时存在三套"方位角"写法：

1. **真倾向方位**（甲方 `Dip Azimuth` 列 / Step3 `Frac_Azimuth`）——真值口径；
2. **罗盘走向**（Step6B/6C 的 `atan2(main[0], main[1])`）——`S = (D-90)%180`；
3. **XY 平面角**（Step7A 脊线 / Step7B 带 PCA / 各 VTK 顶点 `strike=[cosA,sinA]`）。

三者两两相差 90° 或成镜像，混用是 2026-09-22 那轮"成像粉线与 DFN 片走向对不上"的根因。
详见 `太古界/太古界流程梳理与问题记录_20260915.md` §0.37 与
`太古界/太古界产状口径统一施工方案_20260922.md`。

## 用法

```python
from common.orientation_frame import convention as orientation

dip_azimuth, dip = orientation.dip_azimuth_dip_from_normal_depth(normal)   # 向下帧（TIME 为正）
dh, dt = orientation.trace_on_section(dip_azimuth, dip, "XZ")              # 剖面交线方向
```

- 第三维是 `TIME`（向下为正）的场合（Step6B/6C 的 SVD、7A/7B/7C 顶点、Step8 顶点）
  一律用 `*_depth` 变体；
- **业务脚本里不允许再出现裸 `atan2(...)` 算方位**。

单测：`python3 太古界/common/orientation_frame/test_convention.py`
