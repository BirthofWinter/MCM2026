
import matplotlib.pyplot as plt
import numpy as np
import os
import calendar

# 设置风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

def plot_theoretical_comparison():
    # 输出路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', '..', 'data', 'model_images', 'student_union')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    months = np.arange(1, 13)
    month_labels = [calendar.month_abbr[i] for i in months]

    # ==========================================
    # 1. 构建理论数据 (Borealis - 高纬度寒冷地区)
    # ==========================================
    # 单位假设: MJ 或 kWh (相对值更重要)
    
    # Baseline: 北向，差保温，单层玻璃
    # 特点：冬季散热极快(供暖负荷极大)，无法利用太阳能。
    heating_base = np.array([35, 32, 28, 18, 10, 2, 0, 0, 5, 15, 25, 38]) 
    cooling_base = np.array([0,  0,  0,  0,  0,  2, 5, 4, 1, 0,  0,  0]) # 夏季略微制冷
    
    # Optimized: 南向，高保温，三层Low-E玻璃，热质量
    # 特点：冬季由于保温好+被动太阳能，供暖负荷大幅下降。夏季有适当遮阳，制冷负荷可控。
    # 冬季供暖减少约 50-70%
    heating_opt = np.array([18, 15, 10, 4,  1,  0, 0, 0, 0, 3,  10, 20])
    # 夏季制冷略微增加 (因为保温太好，热量散不出去)，但通过夜间通风解决大部分
    cooling_opt = np.array([0,  0,  0,  0,  0,  1, 2, 2, 0, 0,  0,  0])

    # Passive Solar Gain Potential (优化后的南向窗户)
    # 高纬度：春秋季太阳角度低，辐射虽弱但能照进深处。夏季太阳时间长。冬季极短/无。
    solar_gain = np.array([2,  5,  12, 18, 22, 25, 24, 20, 15, 8,  3,  1])

    # ==========================================
    # 2. 绘图
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    width = 0.35
    
    # 绘制 Baseline (左侧，淡色，带斜线)
    p1 = ax1.bar(months - width/2, heating_base, width, label='Heating (Baseline)', color='#e74c3c', alpha=0.4, hatch='//')
    p2 = ax1.bar(months - width/2, cooling_base, width, bottom=heating_base, label='Cooling (Baseline)', color='#3498db', alpha=0.4, hatch='//')
    
    # 绘制 Optimized (右侧，深色，实心)
    p3 = ax1.bar(months + width/2, heating_opt, width, label='Heating (Optimized)', color='#c0392b', alpha=1.0)
    p4 = ax1.bar(months + width/2, cooling_opt, width, bottom=heating_opt, label='Cooling (Optimized)', color='#2980b9', alpha=1.0)
    
    # 绘制 Solar Gain (右轴)
    ax2 = ax1.twinx()
    p5 = ax2.plot(months, solar_gain, color='#f1c40f', marker='o', linewidth=3, markersize=8, label='Passive Solar Gain Received (Optimized)')
    ax2.set_ylabel('Passive Solar Gain (MJ/m²)', fontsize=12, color='#f39c12')
    ax2.tick_params(axis='y', labelcolor='#f39c12')
    ax2.grid(False)
    ax2.set_ylim(0, 30)

    # 装饰
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Monthly HVAC Load (MJ/m²)', fontsize=12)
    ax1.set_title('Annual Energy Performance Comparison: Baseline vs. Passive Solar Design\n(Borealis Student Union)', fontsize=16, pad=20)
    ax1.set_xticks(months)
    ax1.set_xticklabels(month_labels, fontsize=11)
    
    # 辅助线
    ax1.axhline(0, color='black', linewidth=1)
    
    # 自定义图例
    # 收集 handles
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    # 放在顶部
    ax1.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=11)
    
    # 添加注释箭头 - 强调冬季节省
    # 1月数据对比
    base_jan = heating_base[0]
    opt_jan = heating_opt[0]
    ax1.annotate(f'-{int((base_jan-opt_jan)/base_jan*100)}% Heating', 
                 xy=(1 + width/2, opt_jan), xytext=(2.5, opt_jan + 5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black'),
                 fontsize=11, fontweight='bold', color='#c0392b')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'B2_Annual_Energy_Comparison_Theoretical.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison chart generated at: {save_path}")

if __name__ == "__main__":
    plot_theoretical_comparison()
