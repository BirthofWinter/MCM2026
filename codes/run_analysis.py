import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure we can import calculate
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from calculate import BuildingConfig, OverhangStats, NoShading, ThermalSystem

# 设置绘图风格
plt.style.use('ggplot')
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 如果环境没有中文字体可能会报错，暂时用英文通用
plt.rcParams['axes.unicode_minus'] = False 

def load_weather(filepath):
    # Robustly find header
    header_row = 0
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('time'):
                header_row = i
                break
    
    print(f"Detected header at row {header_row} (0-based)")
    
    try:
        df = pd.read_csv(filepath, header=0, skiprows=header_row, engine='python', skipfooter=10)
        # Clean column names
        df.columns = [c.replace(':', '').strip() for c in df.columns]
        
        # Clean data: Convert time and remove invalid rows (footers)
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M', errors='coerce')
        df = df.dropna(subset=['time'])
        
        print(f"Loaded columns: {list(df.columns)}")
        print(f"Loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame()

def run_optimization(location_name, weather_file, lat, is_hot_climate):
    # Parse longitude from file or param
    # Singapore Lon ~ 103.8, Norway ~ 9.0
    if 'singapore' in weather_file.lower():
        lon = 103.8
    else:
        lon = 9.1
        
    print(f"--- Optimizing for {location_name} (Lat: {lat}, Lon: {lon}) ---")
    weather_df = load_weather(weather_file)
    if weather_df.empty:
        print("Weather data empty!")
        return
    
    # 基础建筑配置 (Academic Hall North)
    # 增加 C_in 以模拟家具和建筑热容，这将平滑温度曲线
    config = BuildingConfig(
        C_in=800000.0,      # 增加到 8e5 J/K (相当于约 800kg 空气/家具等效热容)
        Q_internal=500.0,   
        layer_thickness=0.3, 
        wall_area=20.0,     
        window_ratio=0.45,  
        k_wall=0.8,         
        rho_wall=1800.0,
        c_wall=1000.0,
        h_in=8.0,
        h_out=20.0,
        u_window=2.5,       
        tau_window=0.7,     
        k_const_absorb=0.6,
        ventilation_rate=2.0, 
        room_volume=160.0
    )
    
    # 扫描遮阳板长度 L
    l_values = np.linspace(0.0, 3.0, 16) # 0 to 3.0m, extended range
    energy_results = []
    
    best_L = 0.0
    min_cost = float('inf')
    
    # Limit simulation length for speed if needed (e.g. 1 month)
    # But usually 8760 is fast enough for 1D model
    
    for L in l_values:
        strategy = OverhangStats(window_height=1.5, window_width=2.0, wall_azimuth_deg=0, depth_L=L)
        
        sim = ThermalSystem(config, strategy, location_lat=lat, location_lon=lon)
        res_df = sim.simulate(weather_df)
        
        # 统计
        total_cooling = res_df['Q_cooling_load'].sum() # kWh equivalent
        total_heating = res_df['Q_heating_load'].sum() # kWh equivalent
        
        # 成本函数
        if is_hot_climate:
             cost = total_cooling + 0.1 * total_heating # 主要关注制冷
        else:
             cost = total_cooling + total_heating # 两者都关注
        
        energy_results.append({
            'L': L,
            'Cooling': total_cooling,
            'Heating': total_heating,
            'Total': cost
        })
        
        if cost < min_cost:
            min_cost = cost
            best_L = L
            
    results_df = pd.DataFrame(energy_results)
    
    # 绘图 1: 优化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['L'], results_df['Cooling'], label='Cooling Load', marker='o')
    plt.plot(results_df['L'], results_df['Heating'], label='Heating Load', marker='s')
    plt.plot(results_df['L'], results_df['Total'], label='Total Cost Metric', marker='*', linewidth=2)
    plt.axvline(best_L, color='r', linestyle='--', label=f'Best L={best_L:.2f}m')
    plt.xlabel('Overhang Depth L (m)')
    plt.ylabel('Annual Energy Load Metric')
    plt.title(f'Shading Optimization for {location_name}')
    plt.legend()
    plt.grid(True)
    
    output_dir = r'd:\Desktop\美赛\代码\data\model_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, f'optimization_{location_name}.png'))
    plt.close()
    
    print(f"Best L for {location_name}: {best_L}m")
    
    # 运行最佳方案并绘制全年温度/遮挡
    best_strategy = OverhangStats(window_height=1.5, window_width=2.0, wall_azimuth_deg=0, depth_L=best_L)
    final_sim = ThermalSystem(config, best_strategy, location_lat=lat, location_lon=lon)
    final_res = final_sim.simulate(weather_df)
    
    # 绘图 2: 典型日
    plot_days = [172, 355] # June, Dec
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for i, day_num in enumerate(plot_days):
        start_idx = (day_num - 1) * 24
        end_idx = start_idx + 24
        if end_idx > len(final_res): break
        
        day_data = final_res.iloc[start_idx:end_idx]
        ax = axes[i]
        
        ax.plot(day_data['time'].dt.hour, day_data['T_in'], label='Indoor Temp (C)', color='orange', linewidth=2)
        ax.plot(day_data['time'].dt.hour, day_data['T_out'], label='Outdoor Temp (C)', color='blue', linestyle='--')
        
        ax2 = ax.twinx()
        ax2.bar(day_data['time'].dt.hour, day_data['Solar_Flux'], alpha=0.3, color='yellow', label='Incident Flux (W/m2)')
        ax2.plot(day_data['time'].dt.hour, day_data['F_shade']*100, color='gray', linestyle=':', label='Shade %')
        
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Temperature (C)')
        ax2.set_ylabel('Solar Flux / Shade %')
        ax2.set_ylim(0, max(100, day_data['Solar_Flux'].max()*1.2))
        
        season = 'June' if i==0 else 'December'
        ax.set_title(f'{location_name} - {season} (Day {day_num}) - L={best_L:.2f}m')
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'daily_performance_{location_name}.png'))
    plt.close()

if __name__ == "__main__":
    # 配置路径
    singapore_file = r'd:\Desktop\美赛\代码\data\weather\singapore_data.csv'
    norway_file = r'd:\Desktop\美赛\代码\data\weather\norway_data.csv'
    
    # 1. Sungrove (Singapore, Lat ~1.35, Lon ~103.8)
    # 假设 PVGIS 时间是 UTC，新加坡是 UTC+8
    run_optimization('Sungrove', singapore_file, 1.35, is_hot_climate=True)
    
    # 2. Borealis (Norway, Lat ~60.5, Lon ~9.1)
    # Norway is UTC+1
    run_optimization('Borealis', norway_file, 60.5, is_hot_climate=False)
