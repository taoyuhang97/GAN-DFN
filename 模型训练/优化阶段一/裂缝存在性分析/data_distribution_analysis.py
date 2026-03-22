import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import skew, kurtosis
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import seaborn as sns
import xgboost as xgb
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_heatmap(df_out, save_path, title):
    try:
        import seaborn as sns

        plt.figure(figsize=(6, 5))
        sns.heatmap(df_out, annot=True, cmap="viridis", fmt=".3f")
    except ModuleNotFoundError:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(df_out.values, cmap="viridis")
        ax.set_xticks(np.arange(len(df_out.columns)))
        ax.set_yticks(np.arange(len(df_out.index)))
        ax.set_xticklabels(df_out.columns)
        ax.set_yticklabels(df_out.index)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(df_out.shape[0]):
            for j in range(df_out.shape[1]):
                ax.text(j, i, f"{df_out.iloc[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def get_xgb_classifier():
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

# ===================== 路径 =====================

DATA_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\成像测井\裂缝样本"

SAVE_DIR = r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容一\裂缝存在性预测\成像测井数据分布分析"

os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 特征定义 =====================
imaging_well_features = ['AC', 'GR', 'CAL', 'CNL', 'PE', 'DEN', 'CON1', 'GRSL', 'K', 'KTH', 'TH', 'U']
# imaging_well_features = ['DEN', 'CON1', 'GRSL', 'AC', 'GR']
env_imaging_features = os.getenv("EXP_IMAGING_FEATURES")
if env_imaging_features:
    imaging_well_features = json.loads(env_imaging_features)

seis_features = [f"SEIS_{i}" for i in range(63)]
# seis_features = [f"SEIS_{i}" for i in range(3, 63, 7)]
# seis_features = [f"SEIS_TRUE"]

features = seis_features + imaging_well_features

# ===================== 读取数据 =====================
# csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
csv_files = [
    os.path.join(DATA_DIR, "车660-1_sample.csv"),
    os.path.join(DATA_DIR, "车660-2_sample.csv"),
    os.path.join(DATA_DIR, "车662_sample.csv"),
    os.path.join(DATA_DIR, "车663_sample.csv"),
]

df_list = []

for f in csv_files:
    df = pd.read_csv(f)

    df["WellName"] = os.path.basename(f).split("_")[0]

    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

df["FRACTURE_FLAG"] = df["Frac_Azimuth"].notna().astype(int)

df = df[features + ["FRACTURE_FLAG", "WellName"]].dropna()

well_names = df["WellName"].unique()

print("参与分析井：", well_names)


# ===================== 标准化 =====================
# =====================
# 井内标准化
# =====================
# def well_standardize(df, feature_cols):
#     df_out = []
#
#     for w in df["WellName"].unique():
#         sub = df[df["WellName"] == w].copy()
#
#         scaler = StandardScaler()
#
#         sub[feature_cols] = scaler.fit_transform(sub[feature_cols])
#
#         df_out.append(sub)
#
#     df_out = pd.concat(df_out, ignore_index=True)
#
#     return df_out
# df = well_standardize(df, features)
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
well_names = df["WellName"].unique()
X = df[features].values

X_seis = X[:, :63]
X_log = X[:, 63:]

y = df["FRACTURE_FLAG"].values
well = df["WellName"].values


# ===================== 测井分布统计 =====================

def analyze_log_distribution():
    results = []

    for w in well_names:

        idx = well == w

        log = X_log[idx]

        stats = {"Well": w}

        for i, name in enumerate(imaging_well_features):
            curve = log[:, i]

            stats[f"{name}_mean"] = np.mean(curve)
            stats[f"{name}_std"] = np.std(curve)
            stats[f"{name}_skew"] = skew(curve)

        stats["fracture_ratio"] = y[idx].mean()

        results.append(stats)

    df_out = pd.DataFrame(results)

    df_out.to_csv(
        os.path.join(SAVE_DIR, "well_statistics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return df_out


# ===================== 地震振幅统计 =====================

def analyze_seismic_amplitude():
    results = []

    for w in well_names:
        idx = well == w

        seis = X_seis[idx]

        amp = seis.flatten()

        results.append({

            "Well": w,
            "amp_mean": np.mean(amp),
            "amp_std": np.std(amp),
            "amp_skew": skew(amp),
            "amp_kurtosis": kurtosis(amp)

        })

    df_out = pd.DataFrame(results)

    df_out.to_csv(
        os.path.join(SAVE_DIR, "seismic_amplitude.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return df_out


# ===================== 地震空间复杂度 =====================

def analyze_seismic_complexity():
    seis = X_seis.reshape(-1, 3, 3, 7)

    # 空间方差
    spatial_var = np.var(seis, axis=(1,2,3))

    # 时间方差
    temporal_var = np.var(seis, axis=3).mean(axis=(1,2))

    # 梯度
    gx = np.diff(seis, axis=1)
    gy = np.diff(seis, axis=2)

    grad_energy = (
            np.mean(gx ** 2, axis=(1, 2, 3)) +
            np.mean(gy ** 2, axis=(1, 2, 3))
    )

    df_complex = pd.DataFrame({
        "spatial_variance": spatial_var,
        "temporal_variance": temporal_var,
        "gradient_energy": grad_energy,
        "fracture_flag": y,
        "well": well
    })

    save_path = os.path.join(
        SAVE_DIR,
        "seismic_complexity.csv"
    )

    df_complex.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("地震复杂度保存:", save_path)

    return df_complex


# ===================== 井间分布距离 =====================

def compute_well_distance():
    seis_cube = X_seis.reshape(-1, 3, 3, 7)

    seismic_scenarios = {
        "seismic_3x3x7": X_seis,
        "seismic_3x3_plane": seis_cube[:, :, :, 3].reshape(len(seis_cube), -1),
        "seismic_single_amplitude": seis_cube[:, 1, 1, 3].reshape(-1, 1),
        "seismic_center_trace_7": seis_cube[:, 1, 1, :],
    }

    scenarios = {
        "imaging_only": X_log,
    }
    scenarios.update(seismic_scenarios)
    for scenario_name, scenario_features in seismic_scenarios.items():
        scenarios[f"{scenario_name}_plus_imaging"] = np.concatenate(
            [scenario_features, X_log],
            axis=1
        )

    def compute_distance_matrix(feature_matrix):
        matrix = []

        for w1 in well_names:
            idx1 = well == w1
            row = []

            for w2 in well_names:
                idx2 = well == w2
                dists = []

                for i in range(feature_matrix.shape[1]):
                    d = wasserstein_distance(
                        feature_matrix[idx1, i],
                        feature_matrix[idx2, i]
                    )
                    dists.append(d)

                row.append(np.mean(dists))

            matrix.append(row)

        return np.array(matrix)

    save_dir = os.path.join(SAVE_DIR, "well_distance_analysis")
    os.makedirs(save_dir, exist_ok=True)

    summary_records = []

    for scenario_name, feature_matrix in tqdm(
        scenarios.items(),
        desc="计算井间距离"
    ):
        matrix = compute_distance_matrix(feature_matrix)

        df_out = pd.DataFrame(
            matrix,
            index=well_names,
            columns=well_names
        )

        df_out.to_csv(
            os.path.join(save_dir, f"{scenario_name}_distance_matrix.csv"),
            encoding="utf-8-sig"
        )

        plot_heatmap(
            df_out,
            os.path.join(save_dir, f"{scenario_name}_distance_matrix.png"),
            scenario_name
        )

        for i, w1 in enumerate(well_names):
            for j, w2 in enumerate(well_names):
                summary_records.append({
                    "scenario": scenario_name,
                    "well_1": w1,
                    "well_2": w2,
                    "distance": matrix[i, j]
                })

        if scenario_name == "imaging_only":
            df_out.to_csv(
                os.path.join(SAVE_DIR, "well_distance_matrix.csv"),
                encoding="utf-8-sig"
            )

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(
        os.path.join(save_dir, "well_distance_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return summary_df


# ===================== PCA =====================

def plot_pca():
    X_all = np.concatenate([X_seis, X_log], axis=1)

    pca = PCA(n_components=2)

    Z = pca.fit_transform(X_all)

    plt.figure(figsize=(6, 6))

    for w in well_names:
        idx = well == w

        plt.scatter(
            Z[idx, 0],
            Z[idx, 1],
            label=w,
            alpha=0.6
        )

    plt.legend()

    plt.title("PCA Distribution")

    plt.savefig(
        os.path.join(SAVE_DIR, "pca_distribution.png"),
        dpi=300
    )

    plt.close()


# ===================== t-SNE =====================

def plot_tsne():
    X_all = np.concatenate([X_seis, X_log], axis=1)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)

    Z = tsne.fit_transform(X_all)

    plt.figure(figsize=(6, 6))

    for w in well_names:
        idx = well == w

        plt.scatter(
            Z[idx, 0],
            Z[idx, 1],
            label=w,
            alpha=0.6
        )

    plt.legend()

    plt.title("tSNE Distribution")

    plt.savefig(
        os.path.join(SAVE_DIR, "tsne_distribution.png"),
        dpi=300
    )

    plt.close()


# ===================== 裂缝可分性分析 =====================

def analyze_fracture_separability():
    print("\n开始裂缝可分性分析...\n")

    X_all = np.concatenate([X_seis, X_log], axis=1)
    feature_names = seis_features + imaging_well_features
    summary_results = []
    detail_results = []

    for i, name in tqdm(list(enumerate(feature_names)), total=len(feature_names), desc="分析裂缝可分性"):
        per_well_mean_diffs = []
        per_well_abs_mean_diffs = []
        per_well_aucs = []
        per_well_ks_stats = []
        direction_signs = []

        for current_well in well_names:
            idx = well == current_well
            y_well = y[idx]
            x_well = X_all[idx, i]

            fracture_values = x_well[y_well == 1]
            nonfracture_values = x_well[y_well == 0]

            fracture_mean = np.mean(fracture_values) if len(fracture_values) > 0 else np.nan
            nonfracture_mean = np.mean(nonfracture_values) if len(nonfracture_values) > 0 else np.nan
            fracture_std = np.std(fracture_values) if len(fracture_values) > 0 else np.nan
            nonfracture_std = np.std(nonfracture_values) if len(nonfracture_values) > 0 else np.nan

            mean_diff = fracture_mean - nonfracture_mean
            abs_mean_diff = abs(mean_diff) if np.isfinite(mean_diff) else np.nan

            if len(fracture_values) > 5 and len(nonfracture_values) > 5:
                ks_stat, ks_p = ks_2samp(fracture_values, nonfracture_values)
            else:
                ks_stat, ks_p = np.nan, np.nan

            if len(np.unique(y_well)) > 1:
                try:
                    auc = roc_auc_score(y_well, x_well)
                    auc = max(auc, 1 - auc)
                except Exception:
                    auc = np.nan
            else:
                auc = np.nan

            if np.isfinite(mean_diff) and mean_diff != 0:
                direction_signs.append(np.sign(mean_diff))

            if np.isfinite(mean_diff):
                per_well_mean_diffs.append(mean_diff)
            if np.isfinite(abs_mean_diff):
                per_well_abs_mean_diffs.append(abs_mean_diff)
            if np.isfinite(auc):
                per_well_aucs.append(auc)
            if np.isfinite(ks_stat):
                per_well_ks_stats.append(ks_stat)

            detail_results.append({
                "feature": name,
                "well": current_well,
                "fracture_samples": int((y_well == 1).sum()),
                "nonfracture_samples": int((y_well == 0).sum()),
                "fracture_mean": fracture_mean,
                "nonfracture_mean": nonfracture_mean,
                "fracture_std": fracture_std,
                "nonfracture_std": nonfracture_std,
                "mean_diff": mean_diff,
                "abs_mean_diff": abs_mean_diff,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "single_feature_auc": auc
            })

        direction_consistency = np.nan
        if len(direction_signs) > 0:
            direction_consistency = abs(np.mean(direction_signs))

        summary_results.append({
            "feature": name,
            "well_count": len(well_names),
            "valid_auc_well_count": len(per_well_aucs),
            "mean_diff_mean": np.nanmean(per_well_mean_diffs) if len(per_well_mean_diffs) > 0 else np.nan,
            "mean_diff_abs_mean": np.nanmean(per_well_abs_mean_diffs) if len(per_well_abs_mean_diffs) > 0 else np.nan,
            "mean_diff_std": np.nanstd(per_well_mean_diffs) if len(per_well_mean_diffs) > 0 else np.nan,
            "ks_statistic_mean": np.nanmean(per_well_ks_stats) if len(per_well_ks_stats) > 0 else np.nan,
            "single_feature_auc_mean": np.nanmean(per_well_aucs) if len(per_well_aucs) > 0 else np.nan,
            "single_feature_auc_std": np.nanstd(per_well_aucs) if len(per_well_aucs) > 0 else np.nan,
            "direction_consistency": direction_consistency
        })

    df_out = pd.DataFrame(summary_results).sort_values(
        ["single_feature_auc_mean", "direction_consistency", "mean_diff_abs_mean"],
        ascending=[False, False, False]
    )
    detail_df = pd.DataFrame(detail_results)

    save_path = os.path.join(SAVE_DIR, "fracture_separability.csv")
    detail_path = os.path.join(SAVE_DIR, "fracture_separability_by_well.csv")

    df_out.to_csv(save_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("裂缝可分性汇总结果保存：", save_path)
    print("分井可分性结果保存：", detail_path)

    return df_out


# ===================== 裂缝分布可视化 =====================

def plot_fracture_distribution():
    print("绘制裂缝/非裂缝特征分布...")

    X_all = np.concatenate([X_seis, X_log], axis=1)
    feature_names = seis_features + imaging_well_features

    save_dir = os.path.join(SAVE_DIR, "fracture_distribution")
    os.makedirs(save_dir, exist_ok=True)

    for i, name in tqdm(list(enumerate(feature_names)), total=len(feature_names), desc="绘制裂缝分布"):
        fig, axes = plt.subplots(
            len(well_names),
            1,
            figsize=(7, 3.5 * len(well_names)),
            squeeze=False
        )

        plotted_any = False

        for row_idx, current_well in enumerate(well_names):
            ax = axes[row_idx, 0]
            idx = well == current_well
            x_well = X_all[idx, i]
            y_well = y[idx]

            fracture_values = x_well[y_well == 1]
            nonfracture_values = x_well[y_well == 0]

            if len(fracture_values) >= 2:
                sns.kdeplot(
                    fracture_values,
                    label="fracture",
                    fill=True,
                    ax=ax
                )
                plotted_any = True
            elif len(fracture_values) == 1:
                ax.axvline(fracture_values[0], color="tab:orange", linestyle="-", label="fracture")
                plotted_any = True

            if len(nonfracture_values) >= 2:
                sns.kdeplot(
                    nonfracture_values,
                    label="non-fracture",
                    fill=True,
                    ax=ax
                )
                plotted_any = True
            elif len(nonfracture_values) == 1:
                ax.axvline(nonfracture_values[0], color="tab:blue", linestyle="--", label="non-fracture")
                plotted_any = True

            ax.set_title(
                f"{current_well} | fracture={len(fracture_values)} | non-fracture={len(nonfracture_values)}"
            )
            ax.set_xlabel(name)
            ax.set_ylabel("Density")

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                unique = dict(zip(labels, handles))
                ax.legend(unique.values(), unique.keys())
            else:
                ax.text(0.5, 0.5, "样本不足，无法绘制分布", ha="center", va="center", transform=ax.transAxes)

        fig.suptitle(f"{name} by well", y=0.995)
        fig.tight_layout()

        if plotted_any:
            fig.savefig(
                os.path.join(save_dir, f"{name}.png"),
                dpi=200
            )
        plt.close(fig)

    print("分布图已保存")


# ===================== 裂缝 PCA =====================

def plot_fracture_pca():
    X_all = np.concatenate([X_seis, X_log], axis=1)

    pca = PCA(n_components=2)

    Z = pca.fit_transform(X_all)

    plt.figure(figsize=(6, 6))

    plt.scatter(
        Z[y == 0, 0],
        Z[y == 0, 1],
        label="non-fracture",
        alpha=0.5
    )

    plt.scatter(
        Z[y == 1, 0],
        Z[y == 1, 1],
        label="fracture",
        alpha=0.8
    )

    plt.legend()

    plt.title("Fracture PCA Distribution")

    plt.savefig(
        os.path.join(SAVE_DIR, "fracture_pca.png"),
        dpi=300
    )

    plt.close()


def analyze_seismic_window_importance():
    print("\n===== Seismic Window Importance Analysis =====\n")

    X = X_seis
    label = y

    importance_list = []
    fold_results = []

    for test_well in well_names:
        train_idx = well != test_well
        test_idx = well == test_well

        X_train = X[train_idx]
        y_train = label[train_idx]
        X_test = X[test_idx]
        y_test = label[test_idx]

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[:, 1]

        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, pred)
        else:
            auc = np.nan

        fold_results.append({
            "test_well": test_well,
            "samples": len(y_test),
            "fracture_ratio": y_test.mean(),
            "AUC": auc
        })

        importance_list.append(model.feature_importances_)

    importance = np.mean(np.vstack(importance_list), axis=0)

    # 保存重要性表
    df_imp = pd.DataFrame({
        "feature": seis_features,
        "importance": importance
    }).sort_values("importance", ascending=False)

    save_dir = os.path.join(SAVE_DIR, "seismic_window_analysis")
    os.makedirs(save_dir, exist_ok=True)

    df_imp.to_csv(
        os.path.join(save_dir, "feature_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(fold_results).to_csv(
        os.path.join(save_dir, "window_importance_by_well.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # reshape
    imp_cube = importance.reshape(3, 3, 7)

    np.save(
        os.path.join(save_dir, "importance_cube.npy"),
        imp_cube
    )

    # =====================
    # 时间方向重要性
    # =====================

    time_importance = imp_cube.mean(axis=(0, 1))

    plt.figure(figsize=(6, 4))

    plt.plot(range(-3, 4), time_importance, marker="o")

    plt.xlabel("Time Offset")
    plt.ylabel("Importance")
    plt.title("Time Window Importance")

    plt.grid()

    plt.savefig(
        os.path.join(save_dir, "time_importance.png"),
        dpi=300
    )

    plt.close()

    # =====================
    # 空间重要性
    # =====================

    spatial_importance = imp_cube.mean(axis=2)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        spatial_importance,
        annot=True,
        cmap="viridis"
    )

    plt.title("3×3 Spatial Importance")

    plt.savefig(
        os.path.join(save_dir, "spatial_importance.png"),
        dpi=300
    )

    plt.close()

    # =====================
    # 完整窗口
    # =====================

    fig, axes = plt.subplots(1, 7, figsize=(20, 4))

    for t in range(7):
        sns.heatmap(
            imp_cube[:, :, t],
            ax=axes[t],
            cmap="viridis",
            cbar=False
        )

        axes[t].set_title(f"T{t - 3}")

    plt.suptitle("3×3×7 Window Importance")

    plt.savefig(
        os.path.join(save_dir, "full_window_importance.png"),
        dpi=300
    )

    plt.close()

    print("Window importance analysis finished.\n")


def analyze_seismic_feature_scenarios():
    print("\n===== Seismic Feature Scenario Comparison =====\n")

    seis_cube = X_seis.reshape(-1, 3, 3, 7)

    seismic_scenarios = {
        "seismic_3x3x7": X_seis,
        "seismic_3x3_plane": seis_cube[:, :, :, 3].reshape(len(seis_cube), -1),
        "seismic_single_amplitude": seis_cube[:, 1, 1, 3].reshape(-1, 1),
        "seismic_center_trace_7": seis_cube[:, 1, 1, :],
    }

    scenarios = {
        "imaging_only": X_log,
    }

    for name, X_s in seismic_scenarios.items():
        scenarios[name] = X_s
        scenarios[f"{name}_plus_imaging"] = np.concatenate([X_s, X_log], axis=1)

    results = []
    fold_results = []

    save_dir = os.path.join(SAVE_DIR, "seismic_scenario_analysis")
    os.makedirs(save_dir, exist_ok=True)

    for name, X_s in scenarios.items():
        print("训练模型:", name)

        well_aucs = []
        importance_list = []

        for test_well in well_names:
            train_idx = well != test_well
            test_idx = well == test_well

            X_train = X_s[train_idx]
            y_train = y[train_idx]
            X_test = X_s[test_idx]
            y_test = y[test_idx]

            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

            model.fit(X_train, y_train)
            pred = model.predict_proba(X_test)[:, 1]

            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, pred)
            else:
                auc = np.nan

            well_aucs.append(auc)
            importance_list.append(model.feature_importances_)

            fold_results.append({
                "scenario": name,
                "test_well": test_well,
                "samples": len(y_test),
                "fracture_ratio": y_test.mean(),
                "AUC": auc
            })

        auc_array = np.array(well_aucs, dtype=float)
        mean_auc = np.nanmean(auc_array)
        std_auc = np.nanstd(auc_array)

        results.append({
            "scenario": name,
            "feature_dim": X_s.shape[1],
            "AUC_mean": mean_auc,
            "AUC_std": std_auc,
            "uses_imaging": int("plus_imaging" in name or name == "imaging_only"),
            "uses_seismic": int(name != "imaging_only")
        })

        # 保存跨井平均特征重要性
        imp = np.mean(np.vstack(importance_list), axis=0)

        np.save(
            os.path.join(save_dir, f"{name}_importance.npy"),
            imp
        )

    df = pd.DataFrame(results)
    fold_df = pd.DataFrame(fold_results)

    df.to_csv(
        os.path.join(save_dir, "scenario_performance.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    fold_df.to_csv(
        os.path.join(save_dir, "scenario_performance_by_well.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nScenario comparison result:")

    print(df.sort_values("AUC_mean", ascending=False))

    return df


# =====================
# 井交叉验证
# =====================

def well_cross_validation():
    print("\n===== Leave-One-Well-Out CV =====\n")

    X_all = np.concatenate([X_seis, X_log], axis=1)

    results = []

    for test_well in well_names:
        print("测试井:", test_well)

        train_idx = well != test_well
        test_idx = well == test_well

        X_train = X_all[train_idx]
        y_train = y[train_idx]

        X_test = X_all[test_idx]
        y_test = y[test_idx]

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)

        pred = model.predict_proba(X_test)[:, 1]

        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, pred)
        else:
            auc = np.nan

        results.append({

            "test_well": test_well,
            "samples": len(y_test),
            "fracture_ratio": y_test.mean(),
            "AUC": auc

        })

    df = pd.DataFrame(results)

    df.to_csv(
        os.path.join(SAVE_DIR, "well_cv_performance.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n井交叉验证结果：")

    print(df)

    print("Mean AUC:", df["AUC"].mean())

    return df


# =====================
# 中心Trace分析
# =====================

def analyze_center_trace():
    print("\n===== Center Trace Analysis =====\n")

    seis_cube = X_seis.reshape(-1, 3, 3, 7)

    center_trace = seis_cube[:, 1, 1, :]

    results = []

    save_dir = os.path.join(SAVE_DIR, "center_trace_analysis")

    os.makedirs(save_dir, exist_ok=True)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(center_trace, y)

    pred = model.predict_proba(center_trace)[:, 1]

    auc = roc_auc_score(y, pred)

    print("Center Trace AUC:", auc)

    results.append({

        "scenario": "center_trace_7",
        "feature_dim": center_trace.shape[1],
        "AUC": auc

    })

    df = pd.DataFrame(results)

    df.to_csv(
        os.path.join(save_dir, "center_trace_performance.csv"),
        index=False
    )

    return df


# ===================== 主程序 =====================
np.random.seed(42)
print("\n开始数据分布分析...\n")
steps = [
    ("测井统计", analyze_log_distribution),
    ("地震振幅统计", analyze_seismic_amplitude),
    ("地震复杂度", analyze_seismic_complexity),
    ("井间距离", compute_well_distance),
    ("PCA分析", plot_pca),
    ("tSNE分析", plot_tsne),
    ("裂缝可分性", analyze_fracture_separability),
    ("裂缝分布图", plot_fracture_distribution),
    ("裂缝PCA", plot_fracture_pca),
    ("窗口重要性", analyze_seismic_window_importance),
    ("中心Trace分析", analyze_center_trace),
    ("地震场景重要性", analyze_seismic_feature_scenarios),
    ("井交叉验证", well_cross_validation)
]

step_filter = os.getenv("EXP_ANALYSIS_STEP")
if step_filter:
    steps = [(name, func) for name, func in steps if name == step_filter]
    if not steps:
        raise ValueError(f"未找到指定分析步骤: {step_filter}")

for name, func in tqdm(steps, desc="总体分析进度"):
    print(f"\n===== {name} =====")
    func()

print("\n分析完成，结果保存在：", SAVE_DIR)
