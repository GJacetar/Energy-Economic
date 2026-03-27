import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV
import warnings

warnings.filterwarnings('ignore')

# ================= 设置 =================
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="ticks", font='SimSun', font_scale=1.1)

df = pd.read_csv('Final_IRR_Dataset_for_AI_MC.csv')

# 定义特征
continuous_features = ['Annual_PV_Yield_kWh', 'Self_Consume_Rate_%', 'Peak_Valley_Spread', 'Lat', 'Lon',
                       'PV_to_Load_Ratio']
target = 'NPV_Yuan'

# ================= 1. 描述性统计与 VIF 共线性检验 =================
def run_vif_and_desc():
    # 描述性统计
    desc_stats = df[continuous_features + [target]].describe().T
    print("\n--- 表1：核心变量描述性统计 ---")
    print(desc_stats[['count', 'mean', 'std', 'min', 'max']].round(2))

    # VIF 检验 (选取原文表2中的核心变量进行共线性诊断)
    vif_cols = ['Annual_PV_Yield_kWh', 'Lat', 'Peak_Valley_Spread', 'Lon', 'Self_Consume_Rate_%']
    X_vif = df[vif_cols].dropna()
    X_vif_with_const = sm.add_constant(X_vif)

    vif_data = pd.DataFrame()
    vif_data["特征变量"] = X_vif_with_const.columns
    vif_data["VIF_方差膨胀因子"] = [variance_inflation_factor(X_vif_with_const.values, i)
                                    for i in range(X_vif_with_const.shape[1])]

    print("\n--- 表2：VIF 共线性检验结果 ---")
    print(vif_data[vif_data["特征变量"] != "const"].sort_values('VIF_方差膨胀因子', ascending=False).round(3))

# ================= 2. 稳健性检验 =================
def run_algorithm_robustness():
    # 对分类变量进行独热编码 (One-Hot Encoding)
    X_encoded = pd.get_dummies(df[continuous_features + ['Load_Type', 'Grid_Region']], drop_first=True)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    models = {
        "OLS (多元线性回归)": LinearRegression(),
        "Random Forest (随机森林)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=300, depth=6, verbose=0, random_state=42)
    }

    results = []
    print("训练各个基准模型")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({"模型名称": name, "R² (拟合优度)": round(r2, 4), "RMSE": round(rmse, 2)})

    res_df = pd.DataFrame(results)
    print("\n--- 表3：不同机器学习与统计模型的预测精度对比 ---")
    print(res_df.to_string(index=False))

# ================= 3. 因果森林 (Causal Forest) =================
def run_causal_forest():
    # 峰谷价差 (Treatment) 对 NPV (Outcome) 的纯粹因果效应
    Y = df[target].values  # Outcome
    T = df['Peak_Valley_Spread'].values  # Continuous Treatment

    # 混淆变量 (Confounders)：气象条件、地理位置等
    W = df[['Annual_PV_Yield_kWh', 'Lat', 'Lon']].values
    # 异质性特征 (X)：自发自用率、容载比
    X = df[['Self_Consume_Rate_%', 'PV_to_Load_Ratio']].values

    # 配置因果森林 DML
    est = CausalForestDML(model_y=LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                          model_t=LassoCV(cv=3),
                          discrete_treatment=False,
                          n_estimators=200, random_state=42)

    # 拟合因果模型
    est.fit(Y, T, X=X, W=W)

    # 预测异质性处理效应 (CATE - Conditional Average Treatment Effect)
    cate_pred = est.effect(X)
    df['CATE'] = cate_pred

    print("\n不同产业在面对价差变化时的纯因果边际收益 (CATE) 均值：")
    cate_summary = df.groupby('Load_Type')['CATE'].mean()
    print(cate_summary)
    variance = cate_summary.max() - cate_summary.min()
    print(f"最大因果差距为: {variance:.2f} 元")

#     # ======= 绘制因果效应图 =======
#     plt.figure(figsize=(10, 6))
#
#     # 峰谷价差与纯粹因果效应的散点趋势图
#     sns.scatterplot(x=df['Peak_Valley_Spread'], y=cate_pred, hue=df['Load_Type'],
#                     palette='Set1', alpha=0.3, edgecolor=None)
#     sns.regplot(x=df['Peak_Valley_Spread'], y=cate_pred, scatter=False,
#                 color='black', line_kws={"linewidth": 2.5, "linestyle": "--"}, label="全局边际因果趋势")
#
#     plt.title("图 6：基于因果森林(Causal Forest)的峰谷价差异质性因果处理效应(CATE)", fontsize=16, fontweight='bold',
#               pad=15)
#     plt.xlabel("真实峰谷价差 (元/度) - Treatment", fontsize=13, fontweight='bold')
#     plt.ylabel("边际纯因果利润增量 (元) - CATE", fontsize=13, fontweight='bold')
#     plt.grid(True, linestyle=':', alpha=0.6)
#
#     # 标注均值线，展示收敛特性
#     plt.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
#
#     plt.legend(loc='best', fontsize=11)
#
#     ax = plt.gca()
#     for spine in ax.spines.values(): spine.set_visible(True)
#     plt.tight_layout()
#     plt.savefig('Fig6_Causal_Forest_CATE.png', dpi=600, bbox_inches='tight')
#     plt.close()
#
# if __name__ == "__main__":
#     run_vif_and_desc()
#     run_algorithm_robustness()
#     run_causal_forest()