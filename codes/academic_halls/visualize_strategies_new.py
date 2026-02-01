import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ensure we can import calculate
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from calculate import BuildingConfig, OverhangStats, NoShading, ThermalSystem

# Style
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# Paths
weather_file = os.path.join(current_dir, '..', '..', 'data', 'weather', 'singapore_data.csv')
output_dir = os.path.join(current_dir, '..', '..', 'data', 'model_images', 'academic_halls')
if not os.path.exists(output_dir): os.makedirs(output_dir)

def load_weather(filepath, offset=8):
    try:
        header_row = 0
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('time'):
                    header_row = i
                    break
        df = pd.read_csv(filepath, header=0, skiprows=header_row, engine='python', skipfooter=10)
        df.columns = [c.replace(':', '').strip() for c in df.columns]
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M', errors='coerce')
        df = df.dropna(subset=['time'])
        
        # Numeric conversion
        cols = ['Gb(i)', 'Gd(i)', 'Gr(i)', 'H_sun', 'T2m', 'WS10m', 'Int']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def run_analysis():
    print("Running Analysis...")
    df_weather = load_weather(weather_file)
    if df_weather.empty: return

    LAT, LON = 1.35, 103.8
    TZ_OFFSET = 8
    
    # Define Strategies
    # Base
    cfg_base = BuildingConfig(
        Q_internal=500.0, k_wall=0.8, c_wall=1000.0, rho_wall=1800.0,
        u_window=5.8, tau_window=0.85, night_cooling=False
    )
    strat_base = NoShading(2.0, 2.0, 90) # West facing
    
    # Strategies same as before
    strat_shade = OverhangStats(2.0, 2.0, 90, depth_L=1.5)
    
    cfg_glass = BuildingConfig(
        Q_internal=500.0, k_wall=0.8, c_wall=1000.0, rho_wall=1800.0,
        u_window=1.8, tau_window=0.4, night_cooling=False
    )
    
    cfg_vent = BuildingConfig(
        Q_internal=500.0, k_wall=0.8, c_wall=1000.0, rho_wall=1800.0,
        u_window=5.8, tau_window=0.85, night_cooling=True, night_vent_rate=8.0
    )
    
    # Combined
    cfg_comb = BuildingConfig(
        Q_internal=500.0, k_wall=0.8, c_wall=1000.0, rho_wall=1800.0,
        u_window=1.8, tau_window=0.4, night_cooling=True, night_vent_rate=8.0
    )
    
    scenarios = [
        ("Baseline", cfg_base, strat_base),
        ("Shading", cfg_base, strat_shade),
        ("Low-E Glass", cfg_glass, strat_base),
        ("Night Vent", cfg_vent, strat_base),
        ("Combined", cfg_comb, strat_shade)
    ]
    
    results = {}
    
    # Run Simulation for Days 130-132 (3 days)
    # Filter weather first to speed up
    df_weather['doy'] = df_weather['time'].dt.dayofyear
    df_run = df_weather[(df_weather['doy'] >= 120) & (df_weather['doy'] <= 135)].copy()
    # RESET INDEX is crucial for calculate.py which iterates by range(len)
    df_run.reset_index(drop=True, inplace=True)
    
    for name, cfg, strat in scenarios:
        sim = ThermalSystem(cfg, strat, LAT, LON)
        res = sim.simulate(df_run.copy())
        # Apply Timezone
        res['time'] = res['time'] + pd.Timedelta(hours=TZ_OFFSET)
        # Crop to target window
        mask = (res['time'].dt.dayofyear >= 130) & (res['time'].dt.dayofyear < 133)
        results[name] = res.loc[mask].copy()

    # ==========================================
    # 1. Bar Chart: Performance Metrics
    # ==========================================
    metrics = []
    comfort_threshold = 26.0
    
    for name, df in results.items():
        temps = df['T_in']
        avg_t = temps.mean()
        max_t = temps.max()
        # % Time Discomfort
        discomfort_hrs = (temps > comfort_threshold).sum() * (24.0 / len(df) * 3) # approx
        discomfort_pct = (temps > comfort_threshold).mean() * 100
        
        metrics.append({
            'Strategy': name,
            'Avg Temp (°C)': avg_t,
            'Max Temp (°C)': max_t,
            'Discomfort (%)': discomfort_pct
        })
    
    df_metrics = pd.DataFrame(metrics)
    
    # Plot Grouped Bar
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df_metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df_metrics['Avg Temp (°C)'], width, label='Average Temp', color='#3498db')
    bars2 = ax1.bar(x + width/2, df_metrics['Max Temp (°C)'], width, label='Peak Temp', color='#e74c3c')
    
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Passive Cooling Strategy Performance (Heatwave Days)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_metrics['Strategy'])
    ax1.legend(loc='upper right')
    ax1.set_ylim(25, 50)
    
    # Add values
    for b in bars1 + bars2:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 0.2, f'{h:.1f}', ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Strategies_BarChart.png'), dpi=300)
    print("Saved Strategies_BarChart.png")
    
    # ==========================================
    # 2. Pie/Donut Chart: Contribution
    # ==========================================
    # Calculate CDH (Cooling Degree Hours) reduction relative to Baseline
    # CDH = Sum(max(0, T - 26))
    
    base_cdh = ((results['Baseline']['T_in'] - 26).clip(lower=0)).sum()
    reductions = {}
    
    # Approximate individual contribution by comparing single strategies to baseline
    # Note: This is an approximation. Interactions exist.
    # But usually asked to "break down" the benefit.
    
    total_reduction_achieved = base_cdh - ((results['Combined']['T_in'] - 26).clip(lower=0)).sum()
    
    # Individual savings
    cdh_shade = ((results['Shading']['T_in'] - 26).clip(lower=0)).sum()
    cdh_glass = ((results['Low-E Glass']['T_in'] - 26).clip(lower=0)).sum()
    cdh_vent  = ((results['Night Vent']['T_in'] - 26).clip(lower=0)).sum()
    
    s_shade = base_cdh - cdh_shade
    s_glass = base_cdh - cdh_glass
    s_vent  = base_cdh - cdh_vent
    
    print(f"Base CDH: {base_cdh:.2f}")
    print(f"Savings: Shade={s_shade:.2f}, Glass={s_glass:.2f}, Vent={s_vent:.2f}")
    
    # Clip negative savings (if any strategy performed worse)
    s_shade = max(0, s_shade)
    s_glass = max(0, s_glass)
    s_vent  = max(0, s_vent)
    
    # Normalize to sum up to 100% of "Combined Effort" (or just compare relative strength)
    total_individual = s_shade + s_glass + s_vent
    
    slices = [s_shade, s_glass, s_vent]
    labels = ['Shading', 'Low-E Glass', 'Night Vent']
    colors = ['#f1c40f', '#9b59b6', '#2ecc71']
    
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(slices, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, pctdistance=0.85)
    
    # Draw circle for Donut
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title("Contribution to Cooling Load Reduction", fontsize=14)
    plt.text(0, 0, f"Total CDH\nReduction\n{total_reduction_achieved:.0f}", ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Strategies_PieChart.png'), dpi=300)
    print("Saved Strategies_PieChart.png")

    # ==========================================
    # 3. Heatmap
    # ==========================================
    # Prepare Data: Index=Strategy, Columns=Time(Hour 0-72)
    # Resample all to hourly
    
    heatmap_data = []
    strategies_list = ["Baseline", "Shading", "Low-E Glass", "Night Vent", "Combined"]
    
    for s in strategies_list:
        df = results[s].copy()
        # Create a relative hour index 0..N
        df['RelHour'] = np.arange(len(df)) * (30/60.0) # assuming dt=30min or similar, wait.
        # Actually simulate produces variable steps. 
        # Best to just take raw list if length matches, or resample.
        # Let's resample to fixed hourly grid for heatmap
        df = df.set_index('time').resample('H').mean(numeric_only=True)
        # Take first 72 hours
        temps = df['T_in'].values[:72]
        heatmap_data.append(temps)
        
    heatmap_matrix = np.array(heatmap_data)
    
    plt.figure(figsize=(12, 5))
    sns.heatmap(heatmap_matrix, cmap='RdYlBu_r', yticklabels=strategies_list, xticklabels=6, annot=False)
    plt.title("Indoor Temperature Heatmap (72 Hours)", fontsize=14)
    plt.xlabel("Time (Hours)")
    plt.ylabel("Strategy")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Strategies_Heatmap.png'), dpi=300)
    print("Saved Strategies_Heatmap.png")

if __name__ == "__main__":
    run_analysis()
