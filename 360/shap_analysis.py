import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="ticks", font='SimSun', font_scale=1.1)


df = pd.read_csv('Final_IRR_Dataset_for_AI_MC.csv')

feature_cols = [
    'Grid_Region', 'Load_Type', 'Annual_PV_Yield_kWh',
    'PV_to_Load_Ratio', 'Self_Consume_Rate_%', 'Peak_Valley_Spread', 'Lat', 'Lon'
]
cat_features = ['Grid_Region', 'Load_Type']

X = df[feature_cols].copy()
y = df['NPV_Yuan'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
full_pool = Pool(X, cat_features=cat_features)

model = CatBoostRegressor(iterations=600, learning_rate=0.08, depth=6, random_seed=42, verbose=0)
model.fit(train_pool)

print(f"训练完毕，测试集R2得分: {model.score(X_test, y_test):.4f}")

shap_values = model.get_feature_importance(full_pool, type='ShapValues')[:, :-1]

feature_importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nSHAP 特征重要性：")
print(importance_df.to_string(index=False))

feature_name_mapping = {
    'Grid_Region': '防区政策异质性',
    'Load_Type': '产业负荷类型',
    'Annual_PV_Yield_kWh': '光照资源禀赋',
    'PV_to_Load_Ratio': '物理约束：容载比',
    'Self_Consume_Rate_%': '自发自用率',
    'Peak_Valley_Spread': '政策红利：峰谷价差',
    'Lat': '纬度', 'Lon': '经度'
}

X_display = X.copy()
for col in cat_features:
    X_display[col] = X_display[col].astype('category').cat.codes
X_display.rename(columns=feature_name_mapping, inplace=True)

# 图 4.1：全局特征重要性
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_display, show=False)
plt.title("图4.1：基于大样本仿真的光伏收益宏观驱动因素全局归因", fontsize=15, fontweight='bold', pad=20)
ax = plt.gca()
for spine in ax.spines.values(): spine.set_visible(True)
plt.tight_layout()
plt.savefig('Fig4_1_SHAP_Summary.png', dpi=600, bbox_inches='tight')
plt.close()

# 图 4.2：特征依赖图
plt.figure(figsize=(9, 6))
shap.dependence_plot(
    '政策红利：峰谷价差', shap_values, X_display,
    interaction_index='物理约束：容载比', show=False, cmap=plt.get_cmap("RdYlBu_r")
)
plt.title("图4.2：物理容量约束下电价政策阈值的非线性刚性收敛", fontsize=15, fontweight='bold', pad=20)

plt.axvline(0.52, color='black', linestyle='--', linewidth=2, label='电价刚性引爆点 (~0.52元)')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6, color='gray')
ax = plt.gca()
for spine in ax.spines.values(): spine.set_visible(True)
plt.tight_layout()
plt.savefig('Fig4_2_SHAP_Dependence.png', dpi=600, bbox_inches='tight')
plt.close()
