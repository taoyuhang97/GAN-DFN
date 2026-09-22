# -*- coding: utf-8 -*-
"""
ParaView Python 脚本 —— 井轨迹 + 井顶端点 + ASCII 3D Text 井名标签
====================================================================

用途
----
在 ParaView 中加载井轨迹 VTK，并完成三件事：
  1. 显示井轨迹折线；
  2. 在每条轨迹的顶端（T4 顶）放一个小圆点；
  3. 在顶端点上方用 ASCII 3D Text 标注井名（车15 -> Che15）。

说明
----
ParaView 的 "3D Text"（vtkVectorText）只支持 ASCII 字形，所以这里使用
CSV 里的 WellNameASCII 列（车15 -> Che15、车页1导眼 -> CheYe1DaoYan）。
它是纯几何文字、不吃纹理单元，可以避开 vtkTextActor3D 中文渲染触发的
OpenGL 纹理警告（Hardware does not support the number of textures defined）。

运行方法
--------
1. ParaView -> Tools -> Python Shell
2. Load Script... 选择本文件 -> Run

Windows 上最省事的用法（推荐）
------------------------------
把下面 3 个文件复制到同一个文件夹（建议英文/数字路径，如 D:\\dfn_wells\\）：
  label_well_trajectories_paraview.py
  well_trajectories_raw_time.vtk
  well_trajectories_annotations.csv
脚本会自动读取自己同目录下的 VTK 和 CSV，不需要修改任何路径。
注意 VTK 和 CSV 必须成对拷贝（矿区 v1 的 VTK 配矿区 v1 的 CSV，
10km 的 VTK 配 10km 的 CSV）。输入文件本身已经只包含目标范围内的井。

如果想手动指定路径（例如 Linux 服务器 / 自定义目录）
-------------------------------------------------------
修改下方配置区的 WELLS_VTK 与 ANNOTATIONS_CSV 即可。Windows 路径建议用
正斜杠，如 D:/dfn_wells/well_trajectories_raw_time.vtk。

所有 print 输出会同时写入脚本同目录的 label_well_trajectories_log.txt。
"""

import os
import sys
# VTK 模块导入（ParaView 6.2.0 兼容）
try:
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import vtkAppendPolyData, vtkTransformPolyDataFilter
    from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkVectorText
    from vtkmodules.vtkCommonDataModel import vtkPolyData
except ImportError:
    # 备选：尝试旧版导入方式
    try:
        from vtk import vtkTransform, vtkAppendPolyData, vtkTransformPolyDataFilter
        from vtk import vtkSphereSource, vtkVectorText
        from vtk import vtkPolyData
    except ImportError:
        # 再备选：paraview.vtk
        from paraview.vtk import (
            vtkTransform, vtkAppendPolyData, vtkTransformPolyDataFilter,
            vtkSphereSource, vtkVectorText, vtkPolyData
        )

try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # ParaView Python Shell 里没有 __file__ 时
    _SCRIPT_DIR = ""

LOG_PATH = os.path.join(_SCRIPT_DIR or os.getcwd(), "label_well_trajectories_log.txt")

try:
    _ORIGINAL_STDOUT
except NameError:  # 只在第一次运行时保存真正的 stdout/stderr，避免重复运行套娃
    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr


class _TeeWriter(object):
    def __init__(self, log_path):
        self._log = open(log_path, "w", encoding="utf-8")
        self._stdout = _ORIGINAL_STDOUT
        self._stderr = _ORIGINAL_STDERR

    def write(self, text):
        try:
            self._stdout.write(text)
        except Exception:
            pass
        try:
            self._log.write(text)
            self._log.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self._stdout.flush()
        except Exception:
            pass
        try:
            self._log.flush()
        except Exception:
            pass


_TEE = _TeeWriter(LOG_PATH)
sys.stdout = _TEE
sys.stderr = _TEE

print("脚本已启动（__name__=%s），日志文件：%s" % (__name__, LOG_PATH))

# ---------------------------------------------------------------------------
# 配置区
# ---------------------------------------------------------------------------
# 如果脚本同目录下存在 well_trajectories_raw_time.vtk 和
# well_trajectories_annotations.csv，会优先使用同目录文件（Windows 推荐做法）。
# 否则使用下面的 Linux 绝对路径（共享服务器上的默认位置）。
_DEFAULT_WELLS_VTK = r"D:\项目\石油开采\断缝储实验\优化阶段\第二轮优化\成果展示\TIME翻转结果\mine_v1\well_trajectories_raw_time_all.vtk"
_DEFAULT_ANNOTATIONS_CSV = r"D:\项目\石油开采\断缝储实验\优化阶段\第二轮优化\成果展示\TIME翻转结果\mine_v1\well_trajectories_annotations_all.csv"

# 10km v2 版（需要时把上面两个 _DEFAULT_ 路径替换成下面两个）
# _DEFAULT_WELLS_VTK = r"E:\项目\石油项目\断缝储\成果展示\砂砾岩\砂砾岩矿区结果\well_trajectories_raw_time_10km.vtk"
# _DEFAULT_ANNOTATIONS_CSV = r"E:\项目\石油项目\断缝储\成果展示\砂砾岩\砂砾岩矿区结果\well_trajectories_annotations_10km.csv"

def _prefer_local(default_path, file_name):
    candidate = os.path.join(_SCRIPT_DIR, file_name)
    if _SCRIPT_DIR and os.path.exists(candidate):
        return candidate
    return default_path


WELLS_VTK = _prefer_local(_DEFAULT_WELLS_VTK, "well_trajectories_raw_time.vtk")
ANNOTATIONS_CSV = _prefer_local(_DEFAULT_ANNOTATIONS_CSV, "well_trajectories_annotations.csv")

# 井名标签放在井顶端点上方多少毫秒（TWT）
LABEL_Z_OFFSET_MS = 30.0
# 标签高度（世界坐标单位，与 X/Y 米同量级；矿区约 450 比较合适，可按需调）
LABEL_HEIGHT_UNITS = 300.0
# 标签颜色 RGB
LABEL_COLOR = [1.0, 0.82, 0.05]
# 井轨迹折线显示样式
LINE_WIDTH = 3.0
RENDER_LINES_AS_TUBES = True
# 每隔多少口井打印一次进度
PRINT_EVERY = 10
# 井顶端点小球半径（世界坐标单位）
WELL_TOP_POINT_RADIUS = 150.0
# 井顶端点颜色 RGB
WELL_TOP_POINT_COLOR = [1.0, 0.0, 0.0]

# ---------------------------------------------------------------------------
# 井名 -> ASCII 标签映射（如果 CSV 里没有 WellNameASCII 列时兜底用）
# ---------------------------------------------------------------------------
PINYIN_MAP = {
    "\u8f66": "Che",   # 车
    "\u9875": "Ye",    # 页
    "\u5bfc": "Dao",   # 导
    "\u773c": "Yan",   # 眼
    "\u659c": "Xie",   # 斜
    "\u53e4": "Gu",    # 古
    "\u4e95": "Jing",  # 井
}


def ascii_label(name):
    return "".join(PINYIN_MAP.get(ch, ch) for ch in name)


# ---------------------------------------------------------------------------
# 1) 井轨迹顶端的小圆点
# ---------------------------------------------------------------------------
def make_well_top_points(rows, view):
    """
    将所有井顶端点小球合并为单个几何体，仅消耗1个纹理单元。
    """
    print("[井顶端点] 开始，共 %d 口井" % len(rows))
    import paraview.simple as pvs
    
    # 导入 VTK 模块
    try:
        from vtkmodules.vtkFiltersSources import vtkSphereSource
        from vtkmodules.vtkFiltersCore import vtkAppendPolyData
    except ImportError:
        try:
            from vtk import vtkSphereSource, vtkAppendPolyData
        except ImportError:
            from paraview.vtk import vtkSphereSource, vtkAppendPolyData
    
    radius = WELL_TOP_POINT_RADIUS
    all_spheres = []
    count = 0
    
    for row in rows:
        try:
            x = float(row["LabelX"])
            y = float(row["LabelY"])
            z = float(row["LabelZ"])
        except Exception as exc:
            print("[井顶端点] 跳过坐标异常的行：%s（%s）" % (row.get("WellName"), exc))
            continue
        
        # 创建单个球体
        sphere = vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(12)
        sphere.SetPhiResolution(12)
        sphere.SetCenter(x, y, z)
        sphere.Update()
        
        all_spheres.append(sphere.GetOutput())
        count += 1
    
    if not all_spheres:
        print("[井顶端点] 没有成功创建任何球体")
        return 0
    
    # --- 合并所有球体为单个几何体 ---
    print("  [合并] 正在合并 %d 个球体..." % len(all_spheres))
    
    append_filter = vtkAppendPolyData()
    for sphere in all_spheres:
        append_filter.AddInputData(sphere)
    append_filter.Update()
    
    # 将 VTK 数据对象接入 ParaView 管线
    producer = pvs.TrivialProducer()
    producer.GetClientSideObject().SetOutput(append_filter.GetOutput())
  
    rep = pvs.Show(producer, view)
    try:
        rep.DiffuseColor = WELL_TOP_POINT_COLOR
    except Exception:
        pass
    
    print("[井顶端点] 完成：%d 个（已合并为单个几何体）" % count)
    return count


# ---------------------------------------------------------------------------
# 2) ASCII 3D Text 井名标签
# ---------------------------------------------------------------------------
def make_ascii_3d_text(rows, view):
    """
    将所有标签合并为单个几何体，仅消耗1个纹理单元，彻底解决粉屏问题。
    """
    print("[ASCII 标签] 开始，共 %d 口井" % len(rows))
    import paraview.simple as pvs
    
    # 导入 VTK 模块
    try:
        from vtkmodules.vtkCommonTransforms import vtkTransform
        from vtkmodules.vtkFiltersCore import vtkAppendPolyData, vtkTransformPolyDataFilter
        from vtkmodules.vtkFiltersSources import vtkVectorText
    except ImportError:
        try:
            from vtk import vtkTransform, vtkAppendPolyData, vtkTransformPolyDataFilter, vtkVectorText
        except ImportError:
            from paraview.vtk import vtkTransform, vtkAppendPolyData, vtkTransformPolyDataFilter, vtkVectorText
    
    scale = LABEL_HEIGHT_UNITS
    positioned = 0
    failed = 0
    
    # 存储所有文本的几何数据
    all_polydata = []
    
    for row in rows:
        try:
            x = float(row["LabelX"])
            y = float(row["LabelY"])
            z = float(row["LabelZ"]) + LABEL_Z_OFFSET_MS
        except Exception:
            failed += 1
            continue
            
        label = row.get("WellNameASCII") or ascii_label(row["WellName"])
        
        try:
            # 创建单个文本几何
            vector_text = vtkVectorText()
            vector_text.SetText(label)
            vector_text.Update()
            
            # 获取输出并应用变换（平移 + 缩放）
            poly = vector_text.GetOutput()
            
            # 创建变换矩阵
            transform = vtkTransform()
            transform.Identity()
            transform.Translate(x, y, z)
            transform.Scale(scale, scale, scale)
            
            # 应用变换
            transform_filter = vtkTransformPolyDataFilter()
            transform_filter.SetTransform(transform)
            transform_filter.SetInputData(poly)
            transform_filter.Update()
            
            all_polydata.append(transform_filter.GetOutput())
            positioned += 1
            
        except Exception as exc:
            print("  [ASCII 标签] %s 创建失败：%s" % (label, exc))
            failed += 1
    
    if not all_polydata:
        print("[ASCII 标签] 没有成功创建任何标签")
        return 0, 0
    
    # --- 合并所有文本几何为单个数据集 ---
    print("  [合并] 正在合并 %d 个文本几何..." % len(all_polydata))
    
    append_filter = vtkAppendPolyData()
    for poly in all_polydata:
        append_filter.AddInputData(poly)
    append_filter.Update()
    
    # 将 VTK 数据对象接入 ParaView 管线
    producer = pvs.TrivialProducer()
    producer.GetClientSideObject().SetOutput(append_filter.GetOutput())
  
    # 显示合并后的几何
    rep = pvs.Show(producer, view)
    try:
        rep.DiffuseColor = LABEL_COLOR
    except Exception:
        pass
    
    print("[ASCII 标签] 完成：共 %d 个标签，成功 %d 个（已合并为单个几何体）" % (len(rows), positioned))
    return positioned, positioned

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    import csv
    import os

    print("=" * 70)
    print("井轨迹 + 井顶端点 + ASCII 井名标签 脚本开始")
    print("=" * 70)

    try:
        import paraview.simple as pvs
        print("paraview.simple 导入成功")
    except ImportError:
        print("必须在 ParaView 的 Python 环境里运行本脚本（Tools -> Python Shell）")
        return
    try:
        import paraview
        print("ParaView 版本：%s" % getattr(paraview, "__version__", "未知"))
    except Exception:
        pass

    view = pvs.GetActiveViewOrCreate("RenderView")
    try:
        print("RenderView 已就绪：%s" % view.GetXMLLabel())
    except Exception:
        print("RenderView 已就绪")

    # ---- 检查路径与文件 ----
    print("-" * 70)
    print("VTK 路径：%s" % WELLS_VTK)
    print("CSV 路径：%s" % ANNOTATIONS_CSV)
    for tag, path in (("VTK", WELLS_VTK), ("CSV", ANNOTATIONS_CSV)):
        if os.path.exists(path):
            print("  %s 存在，大小 %.1f MB" % (tag, os.path.getsize(path) / 1024.0 / 1024.0))
        else:
            print("  %s 不存在！请检查路径" % tag)
            return

    # ---- 读取 CSV ----
    print("-" * 70)
    with open(ANNOTATIONS_CSV, "r", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    print("CSV 表头：%s" % (list(rows[0].keys()) if rows else "（空）"))
    print("读取到 %d 口井" % len(rows))
    if not rows:
        print("标注 CSV 为空，脚本终止")
        return
    for row in rows:
        row.setdefault("WellNameASCII", ascii_label(row["WellName"]))
    for i in (0, 1, 2, len(rows) - 1):
        r = rows[i]
        print("  示例井 %2d：%-14s LabelX=%s LabelY=%s LabelZ=%s"
              % (i + 1, r["WellName"], r["LabelX"], r["LabelY"], r["LabelZ"]))
    zs = [float(r["LabelZ"]) for r in rows if _safe_float(r.get("LabelZ"))]
    if zs:
        print("井顶端点 Z 范围：%.1f ~ %.1f ms（TWT）" % (min(zs), max(zs)))

    # ---- 显示井轨迹折线 ----
    print("-" * 70)
    well_bounds = None  # 用于保存井轨迹的数据范围
    if os.path.exists(WELLS_VTK):
        reader = pvs.OpenDataFile(WELLS_VTK)
        # 获取井轨迹数据的范围
        try:
            well_bounds = reader.GetDataInformation().GetBounds()
            print("井轨迹数据范围：X [%.1f, %.1f], Y [%.1f, %.1f], Z [%.1f, %.1f]" % well_bounds)
        except Exception as e:
            print("获取井轨迹范围失败：%s" % e)
        disp = pvs.Show(reader, view)
        try:
            disp.LineWidth = LINE_WIDTH
            disp.RenderLinesAsTubes = RENDER_LINES_AS_TUBES
        except Exception:
            pass
        print("已加载井轨迹并显示折线（LineWidth=%s）" % LINE_WIDTH)
    else:
        print("找不到井轨迹 VTK（仅显示标签和井顶点）")

    # ---- 井顶端点 ----
    print("-" * 70)
    make_well_top_points(rows, view)

    # ---- ASCII 井名标签 ----
    print("-" * 70)
    made, positioned = make_ascii_3d_text(rows, view)
    if positioned < made:
        print("警告：只有 %d/%d 个标签成功定位到井顶端点上方，请检查 ParaView 版本并反馈日志"
              % (positioned, made))
    else:
        print("定位确认：%d 个标签均已放到井顶端点上方 %.0f ms 处" % (positioned, LABEL_Z_OFFSET_MS))

def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


# 注意：不用 if __name__ == "__main__" 判断，因为 ParaView Python Shell
# 执行脚本时 __name__ 不一定是 "__main__"，会整段不执行、没有任何输出。
try:
    main()
except Exception:
    import traceback
    print("脚本执行失败，完整报错如下：")
    traceback.print_exc()
