import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import re

plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white", font='SimSun', font_scale=1.1)

print("=" * 60)
print("🌍 模块 4：全国三联空间底图与省级概率密度分布")
print("=" * 60)

df = pd.read_csv('Final_IRR_Dataset_for_AI_MC.csv')


# ----------------- 1. 省级概率密度小提琴图 -----------------
def clean_grid(x):
    """
    省份归一化函数（取消分区，合并为省级）：
    - 内蒙古东部/西部 → 内蒙古
    - 冀北/河北 → 河北
    - 清理冗余字符，统一省级名称
    """
    x = str(x).strip()  # 去除首尾空格

    # 1. 合并内蒙古所有分区为"内蒙古"
    if any(keyword in x for keyword in ['内蒙古', '蒙东', '蒙西', '内蒙古东部', '内蒙古西部']):
        return '内蒙古'

    # 2. 合并河北/冀北为"河北"
    if any(keyword in x for keyword in ['河北', '冀北']):
        return '河北'

    # 3. 其他省份通用规则（可扩展）
    province_mapping = {
        '广东': ['广东'],
        '山东': ['山东'],
        '江苏': ['江苏'],
        '浙江': ['浙江']
    }
    # 匹配通用省份
    for province, keywords in province_mapping.items():
        if any(kw in x for kw in keywords):
            return province

    # 4. 清理冗余字符（国网/电力/公司等）
    clean_pattern = r'国网|电力|公司|有限|省|市|供电局|分局|集团|（国网）|（内蒙古电力）'
    x = re.sub(clean_pattern, '', x)

    # 5. 最终兜底（保留前4个字符，确保名称简洁）
    return x[:4]


df_plot = df.copy()
df_plot['Grid_Region'] = df_plot['Grid_Region'].apply(clean_grid)

# 按NPV中位数排序（省级维度）
order_list = df_plot.groupby('Grid_Region')['NPV_Yuan'].median().sort_values(ascending=False).index

print(f"🥇 全国 NPV 中位数前三名省份: {list(order_list[:3])}")
print(f"💔 全国 NPV 中位数后三名省份: {list(order_list[-3:])}")

# 绘制小提琴图
plt.figure(figsize=(22, 7))
sns.violinplot(
    x='Grid_Region', y='NPV_Yuan', data=df_plot,
    hue='Load_Type', split=False, inner="quartile",
    palette="muted", linewidth=1.2, order=order_list
)
plt.title("图3-2：全国各省级电网工商业光伏 NPV 蒙特卡洛万级样本概率密度分布", fontsize=18, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.ylabel("25年期投资净现值 NPV (元)", fontweight='bold', fontsize=13)
plt.xlabel("")
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax = plt.gca()
# 显示边框并调整样式
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
plt.tight_layout()
plt.savefig('Fig_MC_Violin_Plot.png', dpi=600, bbox_inches='tight')
plt.close()

# ----------------- 2. 全国三联地理空间地图 -----------------
df_city_mean = df.groupby(['Province', 'City', 'Lat', 'Lon']).agg({
    'Annual_PV_Yield_kWh': 'mean',
    'Peak_Valley_Spread': 'mean',
    'NPV_Yuan': 'mean'
}).reset_index()

china_map_url = "https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json"
try:
    china_gdf = gpd.read_file(china_map_url)
    geometry = [Point(xy) for xy in zip(df_city_mean['Lon'], df_city_mean['Lat'])]
    gdf_points = gpd.GeoDataFrame(df_city_mean, geometry=geometry, crs="EPSG:4326")


    def plot_academic_map(ax, base_gdf, points_gdf, column, cmap, title, legend_label):
        # 绘制中国地图底图
        base_gdf.plot(ax=ax, color='#F4F6F7', edgecolor='#B2BABB', linewidth=0.8)
        # 绘制散点（优化图例显示）
        points_gdf.plot(ax=ax, column=column, cmap=cmap, markersize=35,
                        alpha=0.9, edgecolor='black', linewidth=0.3, legend=True,
                        legend_kwds={
                            'label': legend_label,
                            'orientation': "horizontal",
                            'pad': 0.02,
                            'fraction': 0.04,
                            'shrink': 0.8  # 缩小图例避免遮挡
                        })
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_axis_off()


    # 创建三联图
    fig1, axes1 = plt.subplots(1, 3, figsize=(24, 8))
    plot_academic_map(axes1[0], china_gdf, gdf_points, 'Annual_PV_Yield_kWh',
                      'YlOrRd', "图1：中国自然禀赋空间格局", "年均发电量 (kWh)")
    plot_academic_map(axes1[1], china_gdf, gdf_points, 'Peak_Valley_Spread',
                      'RdYlBu_r', "图2：电价政策红利空间分布", "峰谷价差 (元/度)")
    plot_academic_map(axes1[2], china_gdf, gdf_points, 'NPV_Yuan',
                      'viridis', "图3：工商业光伏蒙特卡洛均值全景", "投资净现值 (元)")

    plt.tight_layout()
    plt.savefig('Fig_MC_National_Spatial_Maps.png', dpi=600, bbox_inches='tight')
    plt.close(fig1)
    print("✅ 空间地图渲染完毕，已高保真(600 DPI)保存。")
except Exception as e:
    print(f"⚠️ 底图下载失败或 GeoPandas 报错跳过绘图: {e}")