# -*- coding: utf-8 -*-
"""
ParaView Python 脚本 —— 太古界井轨迹 + 井顶端点 + ASCII 3D Text 井名标签
====================================================================

用途
----
在 ParaView 中加载井轨迹 VTK，并完成三件事：
  1. 显示井轨迹折线；
  2. 在每条目标层段轨迹的最小 TIME 端放一个小圆点；
  3. 在端点上方用 ASCII 3D Text 标注井名（埕北古406 -> ChengBeiGu406）。

说明
----
ParaView 的 "3D Text"（vtkVectorText）只支持 ASCII 字形，所以这里使用
CSV 里的 WellNameASCII 列（埕北古406 -> ChengBeiGu406）。
它是纯几何文字、不吃纹理单元，可以避开 vtkTextActor3D 中文渲染触发的
OpenGL 纹理警告（Hardware does not support the number of textures defined）。

运行方法
--------
1. ParaView -> Tools -> Python Shell
2. Load Script... 选择本文件 -> Run

Windows 上最省事的用法（推荐）
------------------------------
把下面 3 个文件复制到同一个文件夹（建议英文/数字路径，如 D:\\dfn_wells\\）：
  label_well_trajectories_paraview_all.py
  well_trajectories_raw_time.vtk
  well_trajectories_annotations.csv
脚本会自动读取自己同目录下的 VTK 和 CSV，不需要修改任何路径。
注意 VTK 和 CSV 必须是同一次太古界 Step8 导出的成对文件。
当前 5 km demo 只包含埕北古406和埕北古斜405。

如果想手动指定路径（例如 Linux 服务器 / 自定义目录）
-------------------------------------------------------
修改下方配置区的 WELLS_VTK 与 ANNOTATIONS_CSV 即可。Windows 路径建议用
正斜杠，如 D:/dfn_wells/well_trajectories_raw_time.vtk。

所有 print 输出会同时写入脚本同目录的 label_well_trajectories_log.txt。
"""

import os
import sys
# VTK 模块导入（ParaView 6.1/6.2 兼容）
try:
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import vtkAppendPolyData
    from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
    from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkVectorText
except ImportError:
    # 备选：尝试旧版导入方式
    try:
        from vtk import vtkTransform, vtkAppendPolyData, vtkTransformPolyDataFilter
        from vtk import vtkSphereSource, vtkVectorText
    except ImportError:
        # 再备选：paraview.vtk
        from paraview.vtk import (
            vtkTransform, vtkAppendPolyData, vtkTransformPolyDataFilter,
            vtkSphereSource, vtkVectorText
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
# 否则使用下面的 Windows 成果目录绝对路径。
_DEFAULT_WELLS_VTK = r"D:\项目\石油开采\断缝储实验\优化阶段\太古界\成果展示\20260914\well_trajectories_raw_time.vtk"
_DEFAULT_ANNOTATIONS_CSV = r"D:\项目\石油开采\断缝储实验\优化阶段\太古界\成果展示\20260914\well_trajectories_annotations.csv"

def _prefer_local(default_path, file_name):
    candidate = os.path.join(_SCRIPT_DIR, file_name)
    if _SCRIPT_DIR and os.path.exists(candidate):
        return candidate
    return default_path


WELLS_VTK = _prefer_local(_DEFAULT_WELLS_VTK, "well_trajectories_raw_time.vtk")
ANNOTATIONS_CSV = _prefer_local(_DEFAULT_ANNOTATIONS_CSV, "well_trajectories_annotations.csv")

# 井名标签放在井顶端点上方多少毫秒（TWT）
LABEL_Z_OFFSET_MS = 30.0
# 标签高度（世界坐标单位）。太古界拼音井名较长，
# 5 km demo 使用80可避免单个井名几何宽度接近整个区域。
LABEL_HEIGHT_UNITS = 80.0
# 标签颜色 RGB
LABEL_COLOR = [1.0, 0.82, 0.05]
# 井轨迹折线显示样式
LINE_WIDTH = 3.0
RENDER_LINES_AS_TUBES = True
# 每隔多少口井打印一次进度
PRINT_EVERY = 10
# 井顶端点小球半径（世界坐标单位）
WELL_TOP_POINT_RADIUS = 80.0
# 井顶端点颜色 RGB
WELL_TOP_POINT_COLOR = [1.0, 0.0, 0.0]
# True：执行后自动把当前所有可见数据收进画面。
# 如果已经手动调好 DFN 视角且不希望改变相机，改为 False。
RESET_CAMERA = True

# ---------------------------------------------------------------------------
# 井名 -> ASCII 标签映射（如果 CSV 里没有 WellNameASCII 列时兜底用）
# ---------------------------------------------------------------------------
PINYIN_MAP = {
    "\u57d5": "Cheng", # 埕
    "\u5317": "Bei",   # 北
    "\u659c": "Xie",   # 斜
    "\u53e4": "Gu",    # 古
    "\u6869": "Zhuang",# 桩
    "\u6d77": "Hai",   # 海
    "\u4e95": "Jing",  # 井
}


def ascii_label(name):
    return "".join(PINYIN_MAP.get(ch, ch) for ch in name)


def format_bounds(bounds):
    if bounds is None or len(bounds) != 6:
        return "不可用"
    return "X [%.1f, %.1f], Y [%.1f, %.1f], Z [%.1f, %.1f]" % tuple(bounds)


def data_bounds(polydata):
    try:
        return tuple(float(value) for value in polydata.GetBounds())
    except Exception:
        return None


def display_visibility(rep):
    try:
        return int(rep.Visibility)
    except Exception:
        return "unknown"


def point_in_bounds(point, bounds, tolerance=1.0e-3):
    return (bounds[0] - tolerance <= point[0] <= bounds[1] + tolerance
            and bounds[2] - tolerance <= point[1] <= bounds[3] + tolerance
            and bounds[4] - tolerance <= point[2] <= bounds[5] + tolerance)


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
        return 0, None, None
    
    # --- 合并所有球体为单个几何体 ---
    print("  [合并] 正在合并 %d 个球体..." % len(all_spheres))
    
    append_filter = vtkAppendPolyData()
    for sphere in all_spheres:
        append_filter.AddInputData(sphere)
    append_filter.Update()
    
    # 将 VTK 数据对象接入 ParaView 管线
    merged = append_filter.GetOutput()
    producer = pvs.TrivialProducer()
    try:
        pvs.RenameSource("Taigu_WellHeads_Merged", producer)
    except Exception:
        pass
    producer.GetClientSideObject().SetOutput(merged)
  
    rep = pvs.Show(producer, view)
    rep.Visibility = 1
    try:
        rep.Representation = "Surface"
        rep.DiffuseColor = WELL_TOP_POINT_COLOR
        rep.AmbientColor = WELL_TOP_POINT_COLOR
    except Exception:
        pass
    
    print("[井顶端点] 完成：%d 个（已合并为单个几何体）" % count)
    print("  合并井顶范围：%s" % format_bounds(data_bounds(merged)))
    print("  井顶显示状态：Visibility=%s" % display_visibility(rep))
    return count, producer, rep


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
        from vtkmodules.vtkFiltersCore import vtkAppendPolyData
        from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
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
        return len(rows), 0, None, None
    
    # --- 合并所有文本几何为单个数据集 ---
    print("  [合并] 正在合并 %d 个文本几何..." % len(all_polydata))
    
    append_filter = vtkAppendPolyData()
    for poly in all_polydata:
        append_filter.AddInputData(poly)
    append_filter.Update()
    
    # 将 VTK 数据对象接入 ParaView 管线
    merged = append_filter.GetOutput()
    producer = pvs.TrivialProducer()
    try:
        pvs.RenameSource("Taigu_WellNames_Merged", producer)
    except Exception:
        pass
    producer.GetClientSideObject().SetOutput(merged)
  
    # 显示合并后的几何
    rep = pvs.Show(producer, view)
    rep.Visibility = 1
    try:
        rep.Representation = "Surface"
        rep.DiffuseColor = LABEL_COLOR
        rep.AmbientColor = LABEL_COLOR
    except Exception:
        pass
    
    print("[ASCII 标签] 完成：共 %d 个标签，成功 %d 个，失败 %d 个（已合并为单个几何体）"
          % (len(rows), positioned, failed))
    print("  合并井名范围：%s" % format_bounds(data_bounds(merged)))
    print("  井名显示状态：Visibility=%s" % display_visibility(rep))
    return len(rows), positioned, producer, rep

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
    try:
        source_count_before = len(pvs.GetSources())
    except Exception:
        source_count_before = None

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
        if not row.get("WellNameASCII"):
            row["WellNameASCII"] = ascii_label(row["WellName"])
    sample_indices = sorted(set(range(min(3, len(rows)))) | {len(rows) - 1})
    for i in sample_indices:
        r = rows[i]
        print("  示例井 %2d：%-14s LabelX=%s LabelY=%s LabelZ=%s"
              % (i + 1, r["WellName"], r["LabelX"], r["LabelY"], r["LabelZ"]))
    zs = [float(r["LabelZ"]) for r in rows if _safe_float(r.get("LabelZ"))]
    if zs:
        print("井顶端点 Z 范围：%.1f ~ %.1f ms（TWT）" % (min(zs), max(zs)))
    temporary_rows = [row for row in rows if str(row.get("temporary_neighbor_time_depth", "")).strip() == "1"]
    if temporary_rows:
        print("风险提示：%d 口井使用临时邻井时深：%s"
              % (len(temporary_rows), ", ".join(row["WellName"] for row in temporary_rows)))
        print("说明：仅记录风险，本脚本不修改这些井的显示名称。")

    # ---- 显示井轨迹折线 ----
    print("-" * 70)
    well_bounds = None  # 用于保存井轨迹的数据范围
    if os.path.exists(WELLS_VTK):
        reader = pvs.OpenDataFile(WELLS_VTK)
        try:
            pvs.RenameSource("Taigu_WellTrajectories", reader)
        except Exception:
            pass
        reader.UpdatePipeline()
        # 获取井轨迹数据的范围
        try:
            well_bounds = reader.GetDataInformation().GetBounds()
            print("井轨迹数据范围：%s" % format_bounds(well_bounds))
        except Exception as e:
            print("获取井轨迹范围失败：%s" % e)
        disp = pvs.Show(reader, view)
        disp.Visibility = 1
        try:
            disp.Representation = "Surface"
            disp.LineWidth = LINE_WIDTH
            disp.RenderLinesAsTubes = RENDER_LINES_AS_TUBES
            disp.DiffuseColor = [0.0, 0.0, 0.0]
            disp.AmbientColor = [0.0, 0.0, 0.0]
        except Exception:
            pass
        print("已加载井轨迹并显示折线（LineWidth=%s, Visibility=%s）"
              % (LINE_WIDTH, display_visibility(disp)))
        if well_bounds is not None:
            outside = []
            for row in rows:
                try:
                    point = (float(row["LabelX"]), float(row["LabelY"]), float(row["LabelZ"]))
                    # VTK坐标保留6位小数，CSV保留更高精度，
                    # 允许1 mm/ms内的纯写出舍入误差。
                    if not point_in_bounds(point, well_bounds):
                        outside.append(row["WellName"])
                except Exception:
                    outside.append(row.get("WellName", "unknown"))
            if outside:
                print("警告：以下井的标注锚点超出轨迹 bounds：%s" % ", ".join(outside))
            else:
                print("坐标核验：%d 个井顶锚点均在轨迹 bounds 内" % len(rows))
    else:
        print("找不到井轨迹 VTK（仅显示标签和井顶点）")

    # ---- 井顶端点 ----
    print("-" * 70)
    point_count, point_producer, point_rep = make_well_top_points(rows, view)

    # ---- ASCII 井名标签 ----
    print("-" * 70)
    made, positioned, label_producer, label_rep = make_ascii_3d_text(rows, view)
    if positioned < made:
        print("警告：只有 %d/%d 个标签成功生成并定位，请反馈完整日志"
              % (positioned, made))
    else:
        print("定位确认：%d 个标签均已放到井顶端点上方 %.0f ms 处" % (positioned, LABEL_Z_OFFSET_MS))

    # ---- 显式渲染和最终核验 ----
    print("-" * 70)
    if RESET_CAMERA:
        pvs.ResetCamera(view)
        print("已根据当前可见数据重置相机")
    else:
        print("保留现有相机视角（RESET_CAMERA=False）")
    pvs.Render(view)
    try:
        source_count_after = len(pvs.GetSources())
    except Exception:
        source_count_after = None
    added = (source_count_after - source_count_before
             if source_count_before is not None and source_count_after is not None else "unknown")
    print("显式 Render 完成")
    print("最终核验：井数=%d，轨迹对象=1，合并井顶对象=%d，合并井名对象=%d"
          % (len(rows), 1 if point_producer is not None else 0, 1 if label_producer is not None else 0))
    print("Pipeline 对象数：执行前=%s，执行后=%s，本次新增=%s（正常应为3，不随井数增长）"
          % (source_count_before, source_count_after, added))
    print("显示状态：轨迹=%s，井顶=%s，井名=%s"
          % (display_visibility(disp), display_visibility(point_rep), display_visibility(label_rep)))
    print("=" * 70)
    print("太古界井轨迹可视化脚本执行完成")
    print("=" * 70)

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
