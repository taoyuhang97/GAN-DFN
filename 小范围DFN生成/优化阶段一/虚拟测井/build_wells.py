import os
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d, NearestNDInterpolator
from pykrige.ok import OrdinaryKriging
import xgboost as xgb

warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
# 断层曲面数据目录（.dat文件）
FAULT_SURFACE_DIR = r"D:\康宁畅\中国石油\阶段二\按照地层分类\层位"

# 重采样后的地层数据目录（按断层段划分的CSV文件）
ZONE_DATA_DIR = r"D:\康宁畅\中国石油\阶段二\按照地层分类\相对深度重采样数据"

# 原始测井数据目录（可选，用于获取坐标范围）
RAW_WELL_DIR = r"D:\康宁畅\中国石油\阶段二\按照地层分类\原始井数据"

# 输出目录
OUTPUT_DIR = r"D:\康宁畅\中国石油\阶段二\按照地层分类\构建虚拟井\虚拟井预测结果"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型保存目录
MODEL_DIR = r"D:\康宁畅\中国石油\阶段二\按照地层分类\构建虚拟井\训练好的模型"
os.makedirs(MODEL_DIR, exist_ok=True)

# 已训练好的模型文件路径（如果存在，可以直接加载使用）
PRETRAINED_MODEL_FILE = os.path.join(MODEL_DIR, "models_xgboost.pkl")

# ==================== 全局配置 ====================
RESAMPLE_INTERVAL = 0.2  # 重采样间隔 (ms)
INVALID_VALUE = -99999
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ==================== 1. 数据加载模块 ====================

def read_dat_surface(filepath):
    """读取.dat断层曲面文件"""
    data_lines = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    data_lines.append([x, y, z])
                except:
                    continue

    if not data_lines:
        return None
    return pd.DataFrame(data_lines, columns=['X', 'Y', 'Z'])


def get_fault_surface_info(fault_surface_dir):
    """获取所有断层曲面的信息"""
    fault_info = {}
    for file in os.listdir(fault_surface_dir):
        if not file.endswith('.dat'):
            continue

        name = file.replace('.dat', '')
        # 统一括号格式为英文括号，便于匹配
        name_clean = name.replace('（', '(').replace('）', ')')
        filepath = os.path.join(fault_surface_dir, file)
        df = read_dat_surface(filepath)

        if df is not None and len(df) > 0:
            fault_info[name_clean] = {
                'file': filepath,
                'original_name': name,
                'x_min': df['X'].min(),
                'x_max': df['X'].max(),
                'y_min': df['Y'].min(),
                'y_max': df['Y'].max(),
                'z_min': df['Z'].min(),
                'z_max': df['Z'].max(),
                'n_points': len(df)
            }

    print(f"加载断层曲面: {len(fault_info)} 个")
    return fault_info


def get_surface_depth(x, y, surface_name, fault_info):
    """获取某点处某个断层的深度"""
    if surface_name is None:
        return None

    # 统一括号格式
    surface_name_clean = surface_name.replace('（', '(').replace('）', ')')

    # 尝试直接匹配
    if surface_name_clean in fault_info:
        surface = fault_info[surface_name_clean]
    else:
        # 尝试部分匹配
        matched = None
        for name in fault_info.keys():
            if surface_name_clean in name or name in surface_name_clean:
                matched = name
                break
        if matched is None:
            print(f"      警告: 找不到断层 {surface_name}")
            return None
        surface = fault_info[matched]

    filepath = surface['file']
    df = read_dat_surface(filepath)
    if df is None or len(df) == 0:
        return None

    interp = NearestNDInterpolator(df[['X', 'Y']].values, df['Z'].values)
    try:
        z = interp(x, y)
        return float(z) if not np.isnan(z) else None
    except:
        return None


def load_zone_data(zone_name):
    """加载某个断层段的相对深度重采样数据"""
    # 统一括号格式
    safe_name = zone_name.replace('（', '(').replace('）', ')')
    file_path = os.path.join(ZONE_DATA_DIR, f"{safe_name}_relative_depth.csv")
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    return df


def load_all_zones():
    """加载所有断层段的数据"""
    zone_files = [f for f in os.listdir(ZONE_DATA_DIR) if f.endswith('_relative_depth.csv')]
    zones = {}
    for file in zone_files:
        zone_name = file.replace('_relative_depth.csv', '')
        df = load_zone_data(zone_name)
        if df is not None:
            zones[zone_name] = df
    print(f"加载地层段: {len(zones)} 个")
    return zones


# ==================== 2. 地层段识别模块 ====================

def get_zone_by_point(x, y, fault_info, zone_list):
    """根据 (X, Y) 坐标，判断该点穿过哪些地层段"""
    zones_passed = []

    for zone_name in zone_list:
        # 解析地层段名称
        if 'Between_' in zone_name:
            parts = zone_name.replace('Between_', '').split('_')
            if len(parts) >= 2:
                top_name = parts[0]
                bottom_name = parts[1]
                zone_type = 'between'
            else:
                continue
        elif 'Above_' in zone_name:
            top_name = None
            bottom_name = zone_name.replace('Above_', '')
            zone_type = 'above'
        elif 'Below_' in zone_name:
            top_name = zone_name.replace('Below_', '')
            bottom_name = None
            zone_type = 'below'
        else:
            continue

        # 统一括号格式
        top_name_clean = top_name.replace('（', '(').replace('）', ')') if top_name else None
        bottom_name_clean = bottom_name.replace('（', '(').replace('）', ')') if bottom_name else None

        # 检查点是否在断层范围内
        in_top = True
        in_bottom = True

        if top_name_clean and top_name_clean in fault_info:
            top_info = fault_info[top_name_clean]
            if not (top_info['x_min'] <= x <= top_info['x_max'] and
                    top_info['y_min'] <= y <= top_info['y_max']):
                in_top = False
        elif top_name_clean and top_name_clean not in fault_info:
            in_top = False

        if bottom_name_clean and bottom_name_clean in fault_info:
            bottom_info = fault_info[bottom_name_clean]
            if not (bottom_info['x_min'] <= x <= bottom_info['x_max'] and
                    bottom_info['y_min'] <= y <= bottom_info['y_max']):
                in_bottom = False
        elif bottom_name_clean and bottom_name_clean not in fault_info:
            in_bottom = False

        if zone_type == 'between':
            if in_top and in_bottom:
                zones_passed.append((zone_name, top_name, bottom_name, zone_type))
        elif zone_type == 'above':
            if in_bottom:
                zones_passed.append((zone_name, top_name, bottom_name, zone_type))
        elif zone_type == 'below':
            if in_top:
                zones_passed.append((zone_name, top_name, bottom_name, zone_type))

    return zones_passed


# ==================== 3. 训练模型模块 ====================

def train_kriging_for_zone(zone_df, target_col):
    """为单个地层段训练克里金模型"""
    train_data = zone_df.dropna(subset=[target_col])

    if len(train_data) < 10:
        return None

    X = train_data['X'].values.astype(float)
    Y = train_data['Y'].values.astype(float)
    Z = train_data[target_col].values.astype(float)

    try:
        ok = OrdinaryKriging(
            X, Y, Z,
            variogram_model='spherical',
            verbose=False,
            enable_plotting=False
        )
        return ok
    except Exception as e:
        print(f"    克里金训练失败: {e}")
        return None


def train_xgboost_for_zone(zone_df, target_col):
    """为单个地层段训练XGBoost模型"""
    train_data = zone_df.dropna(subset=[target_col])

    if len(train_data) < 10:
        return None, None

    feature_cols = ['X', 'Y', 'RELATIVE_DEPTH']
    X_train = train_data[feature_cols].values.astype(float)
    y_train = train_data[target_col].values.astype(float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbosity=0
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler


def train_all_zones(zones, method='kriging'):
    """训练所有地层段的模型"""
    print(f"\n{'=' * 60}")
    print(f"训练模型 (方法: {method.upper()})")
    print(f"{'=' * 60}")

    models = {}

    for zone_name, zone_df in zones.items():
        print(f"\n  训练地层段: {zone_name}")
        print(f"    数据点数: {len(zone_df)}")
        print(f"    井数: {zone_df['WELL'].nunique()}")

        if method == 'kriging':
            ac_model = train_kriging_for_zone(zone_df, 'AC')
            ac_scaler = None
            gr_model = train_kriging_for_zone(zone_df, 'GR')
            gr_scaler = None
        else:
            ac_model, ac_scaler = train_xgboost_for_zone(zone_df, 'AC')
            gr_model, gr_scaler = train_xgboost_for_zone(zone_df, 'GR')

        if ac_model is not None and gr_model is not None:
            models[zone_name] = {
                'ac_model': ac_model,
                'ac_scaler': ac_scaler,
                'gr_model': gr_model,
                'gr_scaler': gr_scaler,
                'depth_type': zone_df['DEPTH_TYPE'].iloc[0] if len(zone_df) > 0 else 'top'
            }
            print(f"    ✓ 训练成功")
        else:
            print(f"    ✗ 训练失败")

    print(f"\n成功训练: {len(models)}/{len(zones)} 个地层段")
    return models


# ==================== 4. 预测模块 ====================

def predict_one_zone(x, y, zone_name, top_name, bottom_name, zone_type,
                     models, fault_info, method='kriging'):
    """预测单个地层段在 (x, y) 位置的虚拟井曲线"""
    if zone_name not in models:
        return None

    model_info = models[zone_name]

    # 获取该地层段的相对深度范围
    zone_df = load_zone_data(zone_name)
    if zone_df is None:
        return None

    # 获取相对深度范围
    rel_depths = zone_df['RELATIVE_DEPTH'].values
    rel_min = rel_depths.min()
    rel_max = rel_depths.max()

    # 生成严格的0.2间隔相对深度网格
    start_depth = np.ceil(rel_min / RESAMPLE_INTERVAL) * RESAMPLE_INTERVAL
    end_depth = np.floor(rel_max / RESAMPLE_INTERVAL) * RESAMPLE_INTERVAL

    if start_depth > end_depth:
        return None

    n_points = int((end_depth - start_depth) / RESAMPLE_INTERVAL) + 1
    target_depths = np.linspace(start_depth, end_depth, n_points)
    target_depths = np.round(target_depths, 3)

    # 预测每个相对深度点的AC和GR
    ac_preds = []
    gr_preds = []

    for rel_depth in target_depths:
        if method == 'kriging':
            try:
                ac_pred, _ = model_info['ac_model'].execute('points', x, y)
                gr_pred, _ = model_info['gr_model'].execute('points', x, y)
                ac_preds.append(ac_pred[0])
                gr_preds.append(gr_pred[0])
            except:
                ac_preds.append(np.nan)
                gr_preds.append(np.nan)
        else:
            try:
                feature = np.array([[x, y, rel_depth]])
                feature_scaled = model_info['ac_scaler'].transform(feature)
                ac_pred = model_info['ac_model'].predict(feature_scaled)[0]
                gr_pred = model_info['gr_model'].predict(feature_scaled)[0]
                ac_preds.append(ac_pred)
                gr_preds.append(gr_pred)
            except:
                ac_preds.append(np.nan)
                gr_preds.append(np.nan)

    result_df = pd.DataFrame({
        'RELATIVE_DEPTH': target_depths,
        'AC': ac_preds,
        'GR': gr_preds
    })
    result_df = result_df.dropna()
    return result_df


def convert_to_absolute_depth(rel_depth_df, x, y, top_name, bottom_name, zone_type, fault_info):
    """将相对深度转换为绝对深度"""
    if zone_type == 'above':
        bottom_depth = get_surface_depth(x, y, bottom_name, fault_info)
        if bottom_depth is None:
            return None
        abs_df = rel_depth_df.copy()
        abs_df['TIME'] = bottom_depth + abs_df['RELATIVE_DEPTH']
    elif zone_type == 'below':
        top_depth = get_surface_depth(x, y, top_name, fault_info)
        if top_depth is None:
            return None
        abs_df = rel_depth_df.copy()
        abs_df['TIME'] = top_depth + abs_df['RELATIVE_DEPTH']
    else:  # between
        top_depth = get_surface_depth(x, y, top_name, fault_info)
        if top_depth is None:
            return None
        abs_df = rel_depth_df.copy()
        abs_df['TIME'] = top_depth + abs_df['RELATIVE_DEPTH']
    return abs_df


def merge_and_resample_all_zones(segments_abs):
    """合并所有地层的绝对深度数据，重采样到0.2ms间隔"""
    if not segments_abs:
        return None

    combined = pd.concat(segments_abs, ignore_index=True)
    combined = combined.sort_values('TIME')
    combined = combined.groupby('TIME', as_index=False).agg({'AC': 'mean', 'GR': 'mean'})
    combined = combined.sort_values('TIME')

    if len(combined) < 2:
        return combined

    time_min = combined['TIME'].min()
    time_max = combined['TIME'].max()
    start_time = np.floor(time_min / RESAMPLE_INTERVAL) * RESAMPLE_INTERVAL
    end_time = np.ceil(time_max / RESAMPLE_INTERVAL) * RESAMPLE_INTERVAL
    target_times = np.arange(start_time, end_time + RESAMPLE_INTERVAL / 2, RESAMPLE_INTERVAL)

    try:
        interp_ac = interp1d(combined['TIME'], combined['AC'], kind='linear',
                             fill_value='extrapolate', bounds_error=False)
        interp_gr = interp1d(combined['TIME'], combined['GR'], kind='linear',
                             fill_value='extrapolate', bounds_error=False)
        resampled = pd.DataFrame({
            'TIME': target_times,
            'AC': interp_ac(target_times),
            'GR': interp_gr(target_times)
        })
        return resampled
    except:
        return combined


# ==================== 5. 虚拟井生成模块 ====================

def get_coordinate_range(well_dir):
    """从原始井数据获取坐标范围"""
    x_min, x_max = float('inf'), -float('inf')
    y_min, y_max = float('inf'), -float('inf')

    for file in os.listdir(well_dir):
        if not file.endswith('.csv'):
            continue
        filepath = os.path.join(well_dir, file)
        df = pd.read_csv(filepath)
        if 'X' in df.columns and 'Y' in df.columns:
            x_min = min(x_min, df['X'].min())
            x_max = max(x_max, df['X'].max())
            y_min = min(y_min, df['Y'].min())
            y_max = max(y_max, df['Y'].max())

    return x_min, x_max, y_min, y_max


def generate_random_points(n_points, x_min, x_max, y_min, y_max):
    """在指定范围内随机生成N个点"""
    x_coords = np.random.uniform(x_min, x_max, n_points)
    y_coords = np.random.uniform(y_min, y_max, n_points)
    return list(zip(x_coords, y_coords))


def predict_virtual_well(x, y, models, fault_info, all_zones, method='kriging'):
    """预测单个虚拟井的完整曲线"""
    print(f"\n  预测虚拟井: X={x:.2f}, Y={y:.2f}")

    zones_passed = get_zone_by_point(x, y, fault_info, list(all_zones.keys()))
    print(f"    穿过的地层段: {len(zones_passed)} 个")

    if not zones_passed:
        print(f"    警告: 该点不在任何断层范围内")
        return None, None

    segment_results = {}
    segments_abs = []

    for zone_name, top_name, bottom_name, zone_type in zones_passed:
        print(f"      预测地层段: {zone_name}")

        pred_rel_df = predict_one_zone(x, y, zone_name, top_name, bottom_name, zone_type,
                                       models, fault_info, method)

        if pred_rel_df is None or len(pred_rel_df) == 0:
            print(f"        预测失败")
            continue

        segment_results[zone_name] = pred_rel_df
        print(f"        相对深度预测: {len(pred_rel_df)} 个点")

        pred_abs_df = convert_to_absolute_depth(pred_rel_df, x, y, top_name, bottom_name, zone_type, fault_info)

        if pred_abs_df is None:
            print(f"        深度转换失败")
            continue

        segments_abs.append(pred_abs_df[['TIME', 'AC', 'GR']])
        print(f"        绝对深度转换: {len(pred_abs_df)} 个点")

    if not segments_abs:
        return None, None

    combined_df = merge_and_resample_all_zones(segments_abs)

    if combined_df is None:
        return None, None

    print(f"    合并后总点数: {len(combined_df)} 个点")
    return combined_df, segment_results


# ==================== 6. 保存结果模块 ====================

def save_virtual_well_to_excel(well_name, combined_df, segment_results, output_dir, method):
    """保存单个虚拟井的预测结果到Excel文件"""
    output_file = os.path.join(output_dir, f"{well_name}_{method}.xlsx")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        combined_df.to_excel(writer, sheet_name='汇总_完整曲线', index=False)
        for zone_name, rel_df in segment_results.items():
            sheet_name = zone_name[:31]
            rel_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return output_file


def save_all_virtual_wells(wells_results, output_dir, method):
    """保存所有虚拟井的结果"""
    print("\n7. 保存结果...")
    saved_files = []
    for i, ((x, y), (combined_df, segment_results)) in enumerate(wells_results):
        well_name = f"Well_{i + 1}_({x:.0f}_{y:.0f})"
        output_file = save_virtual_well_to_excel(well_name, combined_df, segment_results, output_dir, method)
        saved_files.append(output_file)
        print(f"  已保存: {os.path.basename(output_file)}")
    return saved_files


# ==================== 7. 主函数 ====================

def main():
    """
    主函数
    """
    print("=" * 80)
    print("虚拟测井预测系统")
    print("=" * 80)

    # ==================== 1. 加载数据 ====================
    print("\n1. 加载数据...")
    fault_info = get_fault_surface_info(FAULT_SURFACE_DIR)
    all_zones = load_all_zones()

    if not all_zones:
        print("错误: 没有找到地层段数据")
        return

    # ==================== 2. 选择模式 ====================
    print("\n2. 选择运行模式:")
    print("   1. 训练新模型并预测")
    print("   2. 加载已训练模型并预测")
    mode_choice = input("请输入选择 (1/2): ").strip()

    method = None
    models = None

    if mode_choice == '2':
        # 加载已训练模型
        if os.path.exists(PRETRAINED_MODEL_FILE):
            print(f"\n加载模型: {PRETRAINED_MODEL_FILE}")
            with open(PRETRAINED_MODEL_FILE, 'rb') as f:
                models = pickle.load(f)
            # 从文件名推断方法
            if 'xgboost' in PRETRAINED_MODEL_FILE.lower():
                method = 'xgboost'
            else:
                method = 'kriging'
            print(f"  使用方法: {method.upper()}")
            print(f"  加载成功，共 {len(models)} 个地层段模型")
        else:
            print(f"错误: 找不到模型文件 {PRETRAINED_MODEL_FILE}")
            return
    else:
        # 选择预测方法
        print("\n选择预测方法:")
        print("   1. 克里金插值 (Kriging)")
        print("   2. XGBoost")
        method_choice = input("请输入选择 (1/2): ").strip()
        method = 'kriging' if method_choice == '1' else 'xgboost'
        print(f"  使用方法: {method.upper()}")

        # 训练模型
        print("\n3. 训练模型...")
        models = train_all_zones(all_zones, method=method)

        if not models:
            print("错误: 没有成功训练的模型")
            return

        # 保存模型
        model_file = os.path.join(MODEL_DIR, f"models_{method}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(models, f)
        print(f"\n模型已保存到: {model_file}")

    # ==================== 3. 获取坐标范围 ====================
    x_min, x_max, y_min, y_max = get_coordinate_range(RAW_WELL_DIR)
    print(f"\n坐标范围:")
    print(f"   X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"   Y: [{y_min:.2f}, {y_max:.2f}]")

    # ==================== 4. 选择输入方式 ====================
    print("\n选择输入方式:")
    print("   1. 手动输入单个坐标")
    print("   2. 随机生成N个坐标")
    input_choice = input("请输入选择 (1/2): ").strip()

    points = []
    if input_choice == '1':
        x = float(input("请输入 X 坐标: ").strip())
        y = float(input("请输入 Y 坐标: ").strip())
        points.append((x, y))
    else:
        n = int(input("请输入要生成的虚拟井数量: ").strip())
        points = generate_random_points(n, x_min, x_max, y_min, y_max)
        print(f"  生成 {n} 个随机点")

    # ==================== 5. 预测虚拟井 ====================
    print("\n开始预测虚拟井...")
    wells_results = []
    for i, (x, y) in enumerate(points):
        print(f"\n  预测虚拟井 {i + 1}/{len(points)}")
        if x < x_min or x > x_max or y < y_min or y > y_max:
            print(f"    警告: 点 ({x:.2f}, {y:.2f}) 超出坐标范围")

        combined_df, segment_results = predict_virtual_well(x, y, models, fault_info, all_zones, method)

        if combined_df is not None and segment_results:
            wells_results.append(((x, y), (combined_df, segment_results)))
        else:
            print(f"    预测失败")

    # ==================== 6. 保存结果 ====================
    if wells_results:
        saved_files = save_all_virtual_wells(wells_results, OUTPUT_DIR, method)
        print(f"\n结果已保存到: {OUTPUT_DIR}")
        print(f"共生成 {len(saved_files)} 个Excel文件")
    else:
        print("\n没有成功预测的虚拟井")

    print("\n" + "=" * 80)
    print("完成！")


if __name__ == "__main__":
    main()