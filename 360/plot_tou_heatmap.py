import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df_price = pd.read_csv('电价数据.csv')

def safe_float(val):
    if pd.isna(val) or str(val).strip() in ['无', 'nan', '', 'None']: return np.nan
    try:
        clean_val = re.sub(r'[^\d.]', '', str(val))
        return float(clean_val) if clean_val else np.nan
    except: return np.nan

for col in ['高峰电价(元)', '尖峰电价(元)', '低谷电价(元)', '深谷电价(元)']:
    df_price[col] = df_price[col].apply(safe_float)

def get_spread(row):
    prices = [row['高峰电价(元)'], row['尖峰电价(元)'], row['低谷电价(元)'], row['深谷电价(元)']]
    valid = [p for p in prices if not np.isnan(p)]
    return max(valid) - min(valid) if valid else 0

df_price['Spread'] = df_price.apply(get_spread, axis=1)
df_price['Month'] = df_price['执行月份'].str.extract(r'(\d+)月').astype(int)
df_price['Province'] = df_price['省份'].str.strip()

pivot_spread = df_price.pivot_table(index='Province', columns='Month', values='Spread', aggfunc='mean')
pivot_spread['mean'] = pivot_spread.mean(axis=1)
pivot_spread = pivot_spread.sort_values('mean', ascending=False).drop(columns=['mean'])

plt.figure(figsize=(12, 9))
sns.heatmap(pivot_spread, cmap='RdYlBu_r', linewidths=.5, cbar_kws={'label': '峰谷价差 (元/度)'})
plt.title('图2-4：2025年全国各省12个月工商业分时电价最大峰谷价差全景热力图', fontsize=16, pad=20)
plt.xlabel('执行月份', fontsize=12)
plt.ylabel('省级行政区', fontsize=12)
plt.tight_layout()
plt.savefig('Fig2-4_TOU_Heatmap.png', dpi=600)
print("✅ 图2-4 热力图已生成！")