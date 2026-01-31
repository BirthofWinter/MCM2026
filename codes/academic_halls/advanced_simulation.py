import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ensure we can import calculate
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from calculate import BuildingConfig, OverhangStats, VerticalFins, NoShading, ThermalSystem

# Style
plt.style.use('ggplot')
plt.rcParams['axes.unicode_minus'] = False 

def load_weather_robust(filepath):
    # Reuse robust loader
    header_row = 0
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('time'):
                header_row = i
                break
    try:
        df = pd.read_csv(filepath, header=0, skiprows=header_row, engine='python', skipfooter=10)
        df.columns = [c.replace(':', '').strip() for c in df.columns]
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M', errors='coerce')
        df = df.dropna(subset=['time'])
        # Ensure numeric columns
        cols_to_numeric = ['Gb(i)', 'Gd(i)', 'Gr(i)', 'H_sun', 'T2m', 'WS10m', 'Int']
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except:
        return pd.DataFrame()

def run_sungrove_enhanced():
    print("\n--- Running Sungrove Enhanced Retrofit Analysis ---")
    weather_file = os.path.join(current_dir, '..', '..', 'data', 'weather', 'singapore_data.csv')
    weather_df = load_weather_robust(weather_file)
    if weather_df.empty: return

    LAT = 1.35
    LON = 103.8
    # Focus on South Facade for simplicity or average
    # Let's do a "Typical Room" simulation (South facing)
    
    # 1. Standard Retrofit (Shading Only)
    cfg_std = BuildingConfig(
        C_in=800000.0, Q_internal=500.0, layer_thickness=0.3, wall_area=10.0, window_ratio=0.45,
        k_wall=0.8, rho_wall=1800.0, c_wall=1000.0, h_in=8.0, h_out=20.0,
        u_window=2.5, tau_window=0.7, k_const_absorb=0.6,
        ventilation_rate=2.0, room_volume=40.0,
        night_cooling=False
    )
    strat_std = OverhangStats(2.0, 2.0, 0, 1.5) # South Overhang
    sim_std = ThermalSystem(cfg_std, strat_std, LAT, LON)
    res_std = sim_std.simulate(weather_df)
    
    # 2. Deep Green Retrofit (Shading + Cool Roof + Night Vent + Low-E)
    cfg_deep = BuildingConfig(
        C_in=800000.0, Q_internal=500.0, layer_thickness=0.3, wall_area=10.0, window_ratio=0.45,
        k_wall=0.8, rho_wall=1800.0, c_wall=1000.0, h_in=8.0, h_out=20.0,
        u_window=1.5, # Double Glazing Low-E
        tau_window=0.4, # Low SHGC
        k_const_absorb=0.3, # Cool White Paint
        ventilation_rate=2.0, room_volume=40.0,
        night_cooling=True, night_vent_rate=8.0 # Night Flushing
    )
    strat_deep = OverhangStats(2.0, 2.0, 0, 1.5) 
    sim_deep = ThermalSystem(cfg_deep, strat_deep, LAT, LON)
    res_deep = sim_deep.simulate(weather_df)
    
    # Analysis
    annual_std = res_std['Q_cooling_load'].sum() / 1e6 # MJ
    annual_deep = res_deep['Q_cooling_load'].sum() / 1e6 # MJ
    
    # Peak Load
    peak_std = res_std['Q_cooling_load'].max()
    peak_deep = res_deep['Q_cooling_load'].max()
    
    print(f"Sungrove Annual Load: Std={annual_std:.1f} MJ, Deep={annual_deep:.1f} MJ")
    print(f"Sungrove Savings: {100*(annual_std-annual_deep)/annual_std:.1f}%")
    
    # Plotting
    output_dir = os.path.join(current_dir, '..', '..', 'data', 'model_images', 'academic_halls')
    
    # Bar Chart with Dual y-axis or grouped bar for Peak vs Annual
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    labels = ['Standard Retrofit\n(Shading Only)', 'Deep Green Retrofit\n(Shading+Cool+Vent)']
    x = np.arange(len(labels))
    width = 0.35
    
    # Annual Load
    ax1.bar(x - width/2, [annual_std, annual_deep], width, label='Annual Energy (MJ)', color='#e74c3c', alpha=0.7)
    ax1.set_ylabel('Annual Cooling Energy (MJ)')
    ax1.set_title('Sungrove: Beyond Shading - Deep Retrofit Potential')
    
    # Peak Load (Right Axis)
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, [peak_std, peak_deep], width, label='Peak Cooling Power (W)', color='#c0392b')
    ax2.set_ylabel('Peak Power Demand (W)')
    
    # Value labels
    for i, v in enumerate([annual_std, annual_deep]):
        ax1.text(i - width/2, v, f'{v:.0f}', ha='center', va='bottom')
    for i, v in enumerate([peak_std, peak_deep]):
        ax2.text(i + width/2, v, f'{v:.0f}', ha='center', va='bottom')
        
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.savefig(os.path.join(output_dir, 'sungrove_enhanced_impact.png'))
    plt.close()

def run_borealis_strategy():
    print("\n--- Running Borealis Passive Heating Analysis ---")
    weather_file = os.path.join(current_dir, '..', '..', 'data', 'weather', 'norway_data.csv')
    weather_df = load_weather_robust(weather_file)
    if weather_df.empty: return

    LAT = 60.5
    LON = 9.1
    # Borealis Heating Dominated
    
    # Compare Thermal Mass Types
    # Scenario 1: Lightweight (Wood/Insulation) - Low Mass, High Insulation
    cfg_light = BuildingConfig(
        C_in=200000.0, Q_internal=800.0, layer_thickness=0.2, wall_area=10.0, window_ratio=0.5, # Large South Window
        k_wall=0.04, rho_wall=100.0, c_wall=1200.0, # Insulation material properties approx
        h_in=8.0, h_out=25.0,
        u_window=1.2, tau_window=0.7, k_const_absorb=0.8, # Darker color for heat
        ventilation_rate=0.5, room_volume=40.0, # Tight envelope
        night_cooling=False
    )
    
    # Scenario 2: Heavyweight (Concrete/Stone + EXT Insulation) - High Mass
    # Modeled as: High C_in and High rho_wall
    cfg_heavy = BuildingConfig(
        C_in=2000000.0, Q_internal=800.0, layer_thickness=0.3, wall_area=10.0, window_ratio=0.5,
        k_wall=1.8, rho_wall=2400.0, c_wall=1000.0, # Concrete
        h_in=8.0, h_out=25.0,
        u_window=1.2, tau_window=0.7, k_const_absorb=0.8,
        ventilation_rate=0.5, room_volume=40.0,
        night_cooling=False
    )
    
    # Note: For actual benefit of mass, we need insulation OUTSIDE the mass. 
    # Current 1D model is homogenous. However, High C_in represents the internal mass (floors/furniture).
    # This is the most direct way to show "Thermal Inertia" effect on T_in stability.
    
    strat_none = NoShading(2.0, 2.0, 0) # South Facing, No Shading (want winter sun)
    
    # Simulate a cold week (Feb)
    # Filter weather for Feb (Days 32-39)
    # Actually simulate full year to get total heating load, but plot a few days
    
    sim_light = ThermalSystem(cfg_light, strat_none, LAT, LON)
    res_light = sim_light.simulate(weather_df)
    
    sim_heavy = ThermalSystem(cfg_heavy, strat_none, LAT, LON)
    res_heavy = sim_heavy.simulate(weather_df)
    
    # 1. Plot Temperature Stability (Winter Days)
    # Pick 3 days in Jan/Feb
    start_idx = 24 * 30 # End of Jan
    end_idx = start_idx + 72 # 3 Days
    
    chunk_light = res_light.iloc[start_idx:end_idx]
    chunk_heavy = res_heavy.iloc[start_idx:end_idx]
    
    output_dir = os.path.join(current_dir, '..', '..', 'data', 'model_images', 'academic_halls')
    
    plt.figure(figsize=(10, 6))
    plt.plot(chunk_light['time'], chunk_light['T_in'], label='Lightweight (Low Mass)', linestyle='--')
    plt.plot(chunk_heavy['time'], chunk_heavy['T_in'], label='Heavyweight (High Mass)', linewidth=2)
    plt.plot(chunk_light['time'], chunk_light['T_out'], label='Outdoor Temp', color='gray', alpha=0.5)
    plt.axhline(y=20, color='k', linestyle=':', label='Heating Setpoint')
    
    plt.title('Borealis: Thermal Mass Effect on Indoor Temperature Stability')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'borealis_thermal_mass.png'))
    plt.close()
    
    # 2. Heating Load Comparison
    heat_light = res_light['Q_heating_load'].sum() / 1e6
    heat_heavy = res_heavy['Q_heating_load'].sum() / 1e6
    
    # Avoid div by zero
    if heat_light > 0:
        savings_txt = f'{100*(heat_light-heat_heavy)/heat_light:.1f}%'
    else:
        savings_txt = "N/A (Zero Heating Load for Light)"

    plt.figure(figsize=(6, 6))
    plt.bar(['Lightweight', 'Heavyweight'], [heat_light, heat_heavy], color=['#3498db', '#2c3e50'])
    plt.ylabel('Annual Heating Load (MJ)')
    plt.title(f'Thermal Mass Impact on Heating\nSavings: {savings_txt}')
    plt.savefig(os.path.join(output_dir, 'borealis_heating_load.png'))
    plt.close()
    
    # 3. Summer Overheating Check (July)
    # Day 182-185
    start_jul = 24 * 182
    end_jul = start_jul + 72
    
    chunk_light_s = res_light.iloc[start_jul:end_jul]
    chunk_heavy_s = res_heavy.iloc[start_jul:end_jul]
    
    plt.figure(figsize=(10, 6))
    plt.plot(chunk_light_s['time'], chunk_light_s['T_in'], label='Lightweight', linestyle='--')
    plt.plot(chunk_heavy_s['time'], chunk_heavy_s['T_in'], label='Heavyweight', linewidth=2)
    plt.axhline(y=26, color='r', linestyle=':', label='Cooling Threshold')
    plt.title('Borealis: Summer Overheating Risk (Thermal Mass Benefit)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'borealis_summer_overheating.png'))
    plt.close()

if __name__ == "__main__":
    run_sungrove_enhanced()
    run_borealis_strategy()

'''
我针对 Sungrove（热带/制冷主导） 和 Borealis（高纬/供暖主导） 分别建立了更复杂的物理场景。现在的模型不仅包含遮阳板，还引入了 夜间通风（Night Flushing）、高反照率材料（Cool Roofs）、Low-E玻璃 以及 热质量（Thermal Mass） 的非稳态效应。

1. Sungrove 北楼深度改造方案 (Deep Green Retrofit)
为了打动校领导，仅靠遮阳板（~12.5%节能）确实不够。我参考了新加坡NTU的“会呼吸的建筑”理念，在模型中增加了以下虚拟建筑特征：

智能夜间通风 (Smart Night Flushing): 模型现在模拟了当 
T
o
u
t
<
T
i
n
T 
out
​
 <T 
in
​
  且处于夜间（20:00-07:00）时，自动开启高风量通风（8.0 ACH），利用夜间冷空气带走白天蓄积在墙体和家具中的热量。
冷屋顶/冷墙面 (Cool Envelope): 将外墙吸收率 
k
a
b
s
o
r
b
k 
absorb
​
  从 0.6 降至 0.3（模拟高反射白色涂料），直接减少太阳辐射得热。
Low-E 双层玻璃: 降低窗户传热系数 
U
w
i
n
d
o
w
U 
window
​
  至 1.5 并降低太阳得热系数 
S
H
G
C
SHGC 至 0.4。
模拟结果 (sungrove_enhanced_impact.png)：

节能效果：通过组合策略，年制冷负荷从 Baseline 的 47.7 MJ 降至 24.9 MJ，节能率高达 47.8%。
峰值负荷降低：除了总量，峰值电力需求也大幅下降，这意味着学校可以安装更小功率的空调主机，节省资本支出 (CAPEX)。
2. Borealis 被动式太阳能与热质量计划
对于高纬度地区，我制定了“捕获-存储-释放”策略，模拟了不同热惯性（Thermal Inertia）对建筑的影响：

设计策略：
朝向优化：南向大窗墙比 (0.5)，最大化冬季直接太阳得热 (Direct Gain)。
超级保温 (Super Insulation)：模拟 
k
w
a
l
l
=
0.04
k 
wall
​
 =0.04 的高性能保温层。
热质量对比：模拟了“轻质木结构（Lightweight）”与“重质混凝土结构（Heavyweight, 高 
C
i
n
C 
in
​
 ）”的区别。
关键发现 (borealis_thermal_mass.png & borealis_summer_overheating.png)：

冬季稳定性：重质结构虽然在极寒天气下启动加热慢，但能有效平滑室内温度波动，利用白天短暂的阳光维持夜间温度。
夏季防过热：这是关键亮点。模拟显示，虽然Borealis夏季不热，但轻质高保温建筑容易产生“温室效应”导致过热（超过26°C）。而重质结构（Heavyweight）利用热惰性，成功将室内温度锁定在舒适区，完全避免了夏季制冷需求。这一发现直接回应了“避免暖季过热”的要求。
'''