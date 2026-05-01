import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import glob
import numpy as np
import os
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay,roc_curve, auc
import matplotlib.pyplot as plt
from xgboost import plot_importance
from imblearn.over_sampling import SMOTE

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ===== 1. 读取已标记样本 =====
# data_dir = r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/成像测井/裂缝存在样本"

# 使用glob查找所有csv文件
# csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

# 当前使用指定测井作为样本
csv_files = [r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/成像测井/裂缝样本/车151HF_sample.csv"]

# 创建一个空的列表，用于存储每口井的数据
df_list = []

# 逐个加载文件
for file in csv_files:
    df = pd.read_csv(file)
    # 可选：增加井的来源标签
    df['WellName'] = os.path.basename(file).split('.')[0]  # 从文件名中提取井名
    df_list.append(df)

# 合并所有井的数据为一个总表
all_samples_df = pd.concat(df_list, ignore_index=True)

print(all_samples_df.shape)
print(all_samples_df.head())

# ===== 2. 准备特征和标签 =====
features = ['SEIS_TRUE'] + [f'SEIS_{i}' for i in range(63)] + ['AC', 'GR']
# features = ['SEIS_TRUE'] + [f'SEIS_{i}' for i in range(3, 63, 7)] + ['AC', 'GR']
# features = ['SEIS_TRUE'] + ['AC', 'GR']
label = 'FRACTURE_FLAG'

df = all_samples_df.copy()

df['FRACTURE_FLAG'] = df['Frac_Azimuth'].notna().astype(int)

df_model = df[features + ['FRACTURE_FLAG']].dropna(subset=features)

X = df_model[features]
y = df_model['FRACTURE_FLAG']

# ===== 3. 标准化（可选） =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== 4. 拆分训练/验证集（可选） =====
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# ===== 4.1 过采样处理（只对训练集做）=====
print(f"过采样前训练集类别分布: {np.bincount(y_train)}")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"过采样后训练集类别分布: {np.bincount(y_train_res)}")

# ===== 5. 训练 XGBoost 模型 =====
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=pos_weight,
    random_state=42
)


# model.fit(X_train, y_train)
model.fit(X_train_res, y_train_res)

# ===== 6. 可选：评估模型 =====
val_acc = model.score(X_val, y_val)
print(f"验证集准确率: {val_acc:.4f}")

y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["无裂缝", "有裂缝"])
disp.plot(cmap=plt.cm.Blues)
plt.title("XGBoost 混淆矩阵")
plt.savefig('XGBoost 混淆矩阵.png')
plt.show()

y_prob = model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC 曲线')
plt.legend(loc='lower right')
plt.savefig('XGBoost ROC 曲线.png')
plt.show()

# 如果你的特征是 DataFrame 形式，可以直接获取列名
feature_names = X.columns  # 假设你的特征是 X
# 获取模型中特征的重要性
importances = model.feature_importances_

# 创建 DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# 按重要性排序
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)

# 可视化特征重要性
# 简化的分组特征重要性可视化
feature_groups = {
    '当前点位地震强度': ['SEIS_TRUE'],
    '周边地震强度(平均)': [f'SEIS_{i}' for i in range(63)],
    'GR属性': ['GR'],
    'AC属性': ['AC']
}

# 计算分组重要性
group_importance = []
for group_name, features in feature_groups.items():
    if group_name == '周边地震强度(平均)':
        imp = importance_df[importance_df['Feature'].isin(features)]['Importance'].mean()
    else:
        imp = importance_df[importance_df['Feature'].isin(features)]['Importance'].values[0]
    group_importance.append((group_name, imp))

# 创建DataFrame并排序
group_df = pd.DataFrame(group_importance, columns=['特征组', '重要性']).sort_values('重要性')

# 绘制水平条形图
plt.figure(figsize=(10, 5))
bars = plt.barh(group_df['特征组'], group_df['重要性'], color='skyblue')
plt.xlabel('特征重要性')
plt.title('分组特征重要性')

# 添加数值标签
for bar, value in zip(bars, group_df['重要性']):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{value:.4f}', va='center')

plt.tight_layout()
plt.savefig('简化分组特征重要性.png', dpi=300, bbox_inches='tight')
plt.show()

# plt.figure(figsize=(10, 20))
# plt.barh(importance_df['Feature'], importance_df['Importance'])
# plt.gca().invert_yaxis()  # 让最高的重要性在上方
# plt.xlabel("Feature Importance")
# plt.title("XGBoost Feature Importance")
# plt.tight_layout()
# plt.savefig('XGBoost Feature Importance.png')
# plt.show()

# ===== 7. 保存模型与 scaler =====
model_dir = r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝存在性预测/XGboost/模型"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "fracture_xgb_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "fracture_scaler.pkl"))
joblib.dump(features, os.path.join(model_dir, "fracture_features.pkl"))

print("模型保存完毕 ✅")