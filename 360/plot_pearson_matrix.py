import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('Final_IRR_Dataset_for_AI_MC.csv')
cols = ['Annual_PV_Yield_kWh', 'Self_Consume_Rate_%', 'Peak_Valley_Spread', 'PV_to_Load_Ratio', 'Lat', 'Lon', 'NPV_Yuan']
names = ['光伏年发电总量', '自发自用率', '峰谷价差', '物理容载比', '纬度', '经度', '全生命周期净现值(NPV)']
corr = df[cols].corr()
corr.columns = names
corr.index = names

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('图2-5：工商业光伏核心特征变量Pearson相关性检验矩阵', fontsize=15, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('Fig2-5_Pearson_Matrix.png', dpi=600)
print("✅ 图2-5 Pearson相关性矩阵图已生成！")