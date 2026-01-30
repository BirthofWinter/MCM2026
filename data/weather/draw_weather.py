import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ---------------------------------------------------------
# 1. 数据加载与预处理函数
# ---------------------------------------------------------
def load_and_clean(filepath):
    """加载PVGIS CSV数据，跳过页眉页脚并转换时间"""
    # 注意：如果你的CSV格式不同，可能需要调整skiprows（通常为16-20行）
    df = pd.read_csv(filepath, skiprows=8, skipfooter=12, engine='python')
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
    return df

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 请确保文件名与你下载的文件一致
try:
    df_sg = load_and_clean(os.path.join(current_dir, 'singapore_data.csv'))  # 暖色系城市：新加坡
    df_no = load_and_clean(os.path.join(current_dir, 'norway_data.csv'))       # 寒色系城市：奥斯陆
except FileNotFoundError:
    print("错误：未找到CSV文件，请检查文件名是否匹配。")
    exit()

# ---------------------------------------------------------
# 2. 定义典型日对比绘图函数（整合建议 1, 2, 3）
# ---------------------------------------------------------
def plot_solstice_comparison(day_str, title_label, filename):
    # 筛选夏至或冬至
    sg_day = df_sg[df_sg['time'].dt.strftime('%m%d') == day_str].reset_index()
    no_day = df_no[df_no['time'].dt.strftime('%m%d') == day_str].reset_index()
    
    hours = range(len(sg_day))

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- 建议3：色彩选择 (SG暖色, NO寒色) ---
    # 绘制直接辐射 (Gb(i))
    ax1.fill_between(hours, sg_day['Gb(i)'], color='orange', alpha=0.2)
    line1, = ax1.plot(hours, sg_day['Gb(i)'], color='darkorange', lw=2.5, label='Singapore Radiation')
    
    ax1.fill_between(hours, no_day['Gb(i)'], color='skyblue', alpha=0.2)
    line2, = ax1.plot(hours, no_day['Gb(i)'], color='dodgerblue', lw=2.5, label='Norway Radiation')

    # --- 建议2：标准单位标注 ---
    ax1.set_xlabel('Time of Day (Hours)', fontsize=12)
    ax1.set_ylabel('Direct Beam Radiation $[W/m^2]$', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 23)
    ax1.set_ylim(0, 1100)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 绘制气温 (T2m) - 次坐标轴
    ax2 = ax1.twinx()
    line3, = ax2.plot(hours, sg_day['T2m'], color='red', linestyle='--', lw=2, label='Singapore Temp')
    line4, = ax2.plot(hours, no_day['T2m'], color='blue', linestyle='--', lw=2, label='Oslo Temp')
    
    ax2.set_ylabel('Air Temperature $[^\circ C]$', fontsize=12, fontweight='bold')
    ax2.set_ylim(-20, 45)

    # 合并图例
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

    plt.title(f'Climate Dynamics Comparison: {title_label}', fontsize=15, pad=20)
    
    # --- 建议1：高清保存图片 (300 DPI) ---
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    print(f"已保存图表: {filename}.png")
    # plt.show()

# ---------------------------------------------------------
# 3. 绘制全年月平均气温对比（辅助模型论证）
# ---------------------------------------------------------
def plot_monthly_analysis():
    df_sg['month'] = df_sg['time'].dt.month
    df_no['month'] = df_no['time'].dt.month
    
    m_temp = pd.DataFrame({
        'Singapore (Sungrove)': df_sg.groupby('month')['T2m'].mean(),
        'Norway (Borealis)': df_no.groupby('month')['T2m'].mean()
    })

    plt.figure(figsize=(12, 6))
    # 暖色/寒色条形图
    m_temp.plot(kind='bar', color=['salmon', 'lightskyblue'], ax=plt.gca(), edgecolor='black', alpha=0.8)
    
    # 舒适区基准线
    plt.axhline(22, color='green', linestyle='--', lw=1.5, label='Human Comfort Zone ($22^\circ C$)')
    
    plt.title('Monthly Mean Temperature vs. Comfort Requirement', fontsize=14)
    plt.ylabel('Temperature $[^\circ C]$', fontsize=12)
    plt.xlabel('Month', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 高清保存
    plt.savefig('monthly_temp_comparison.png', dpi=300, bbox_inches='tight')
    print("已保存图表: monthly_temp_comparison.png")
    # plt.show()

# ---------------------------------------------------------
# 运行绘图
# ---------------------------------------------------------
# 0621 = 夏至, 1221 = 冬至
plot_solstice_comparison('0621', 'Summer Solstice (June 21st)', 'comparison_summer')
plot_solstice_comparison('1221', 'Winter Solstice (December 21st)', 'comparison_winter')
plot_monthly_analysis()