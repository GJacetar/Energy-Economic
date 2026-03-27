import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ================= 顶刊图表规范设置 (600 DPI) =================
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="ticks", font='SimSun', font_scale=1.1)


def plot_perfect_alignment(data, ax, color, title, label, target_irr=20.0):
    """
    核心逻辑：确保曲线、交点、垂直线使用同一套数学底稿，消除视觉偏差
    """
    # 1. 数据预处理
    data = data.sort_values('Peak_Valley_Spread')
    x = data['Peak_Valley_Spread'].values
    y = data['IRR_%'].values

    # 2. 统一使用 statsmodels 进行 LOWESS 平滑 (frac=0.3)
    lowess = sm.nonparametric.lowess(y, x, frac=0.3)
    x_smooth = lowess[:, 0]
    y_smooth = lowess[:, 1]

    # 3. 绘制原始散点 (万级数据降透明度)
    ax.scatter(x, y, alpha=0.03, color=color, s=10, label='仿真样本点')

    # 4. 绘制【同一套底稿】生成的平滑曲线
    ax.plot(x_smooth, y_smooth, color='black', linewidth=2.5, label='LOWESS拟合趋势')

    # 5. 精准计算数学交点 (使用线性插值找到 y=target_irr 对应的 x)
    # 因为 LOWESS 结果是单调增的，直接插值最精准
    cross_x = np.interp(target_irr, y_smooth, x_smooth)

    # 6. 绘制辅助线和交点点位
    ax.axhline(target_irr, color='black', linestyle='--', linewidth=1.2)
    ax.axvline(cross_x, color=color, linestyle='-.', linewidth=2)
    ax.plot(cross_x, target_irr, 'ko', markersize=8, zorder=5)

    # 7. 控制台输出精准数值，供你校对论文文字
    print(f"[{label}] 物理约束测算结果：")
    print(f"   - 样本数量: {len(data)}")
    print(f"   - 平均自发自用率: {data['Self_Consume_Rate_%'].mean():.2f}%")
    print(f"   - 20% IRR 对应的【精准价差阈值】: {cross_x:.4f} 元/度")

    # 8. 图表美化
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("2025年各省峰谷价差 (元/度)", fontsize=13)
    ax.set_ylabel("项目内部收益率 IRR (%)", fontsize=13)
    ax.text(cross_x + 0.02, target_irr - 2, f"阈值: {cross_x:.3f}元",
            color=color, fontweight='bold', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)


# ================= 主程序 =================
print("=" * 60)
print("🚀 启动 Fig 5 高清重构程序：消除曲线与阈值的视觉偏差")
print("=" * 60)

try:
    df = pd.read_csv('Final_IR_Dataset_for_AI_MC.csv')  # 注意确认文件名是否带那个多出来的'R'
except:
    df = pd.read_csv('Final_IRR_Dataset_for_AI_MC.csv')

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# 产业 A
plot_perfect_alignment(
    df[df['Load_Type'] == 'Heavy_Industry'], axes[0],
    '#BC3C29', "图A: 重工业收益阈值收敛特征", "重工业"
)

# 产业 B
plot_perfect_alignment(
    df[df['Load_Type'] == 'Commercial'], axes[1],
    '#E18727', "图B: 商业综合体收益阈值收敛特征", "商业"
)

# 产业 C
plot_perfect_alignment(
    df[df['Load_Type'] == 'Daytime_Mfg'], axes[2],
    '#0072B5', "图C: 轻工业/物流园收益阈值收敛特征", "轻工业"
)

plt.suptitle("图5：基于 2025 年高频时序仿真的产业投资阈值刚性收敛实证 (600 DPI)",
             fontsize=20, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('Fig5_Final_Perfect_Alignment.png', dpi=600, bbox_inches='tight')
plt.close()

print("=" * 60)
print("📁 终极高清图表已生成: Fig5_Final_Perfect_Alignment.png")
print("⚠️ 请复制上述打印的【精准价差阈值】到论文正文中，确保图文一致。")
print("=" * 60)