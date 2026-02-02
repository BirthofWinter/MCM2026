import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
import os
from matplotlib.gridspec import GridSpec

# Ensure we can import calculate
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from calculate import BuildingConfig, OverhangStats, VerticalFins, NoShading, ThermalSystem

# Visual Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
# Colors
c_base = '#95a5a6'
c_shade = '#3498db'
c_glass = '#9b59b6'
c_vent = '#2ecc71'
c_comb = '#e74c3c'

c_heat = '#e67e22'
c_cool = '#3498db'

def load_weather_robust(filepath, timezone_offset=0, apply_offset_to_cols=False):
    # Robustly find header
    header_row = 0
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('time'):
                    header_row = i
                    break
        df = pd.read_csv(filepath, header=0, skiprows=header_row, engine='python', skipfooter=10)
        df.columns = [c.replace(':', '').strip() for c in df.columns]
        
        # PVGIS time is usually UTC. We convert to datetime naive first.
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M', errors='coerce')
        df = df.dropna(subset=['time'])
        
        # CRITICAL FIX: Do NOT apply offset to 'time' column here if it is used for Physics Simulation.
        # Physics engine (calculate.py) expects UTC.
        # We only store the offset for later use or apply if explicitly asked.
        if apply_offset_to_cols:
            df['time'] = df['time'] + pd.Timedelta(hours=timezone_offset)
        
        # Ensure numeric
        cols_to_numeric = ['Gb(i)', 'Gd(i)', 'Gr(i)', 'H_sun', 'T2m', 'WS10m', 'Int']
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

def run_sungrove_part1():
    print("\n=== Running Part 1: Sungrove (Academic Halls) ===")
    
    # Paths
    weather_file = os.path.join(current_dir, '..', '..', 'data', 'weather', 'singapore_data.csv')
    output_dir = os.path.join(current_dir, '..', '..', 'data', 'model_images', 'academic_halls')
    csv_dir = os.path.join(current_dir, '..', '..', 'data', 'csv', 'academic_halls')
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(csv_dir): os.makedirs(csv_dir)

    # Singapore is UTC+8. We LOAD UTC for simulation, but apply offset later.
    TZ_OFFSET = 8
    weather_df = load_weather_robust(weather_file, timezone_offset=TZ_OFFSET, apply_offset_to_cols=False)
    if weather_df.empty:
        print("Error: Sungrove weather data missing.")
        return

    LAT = 1.35
    LON = 103.8

    # ---------------------------------------------------------
    # 1. Four Faces Heat Map (Window Heat Gain Comparison)
    # ---------------------------------------------------------
    print("Generating 1. Solar Heatmaps (4 Faces)...")
    
    # North(180), East(-90), South(0), West(90)
    # Correct Azimuths for solar_geometry.py (0=South, +90=West)
    orientations = {
        'North': 180,
        'East': -90, 
        'South': 0,
        'West': 90
    }
    
    # Use a dummy config just for flux calculation
    dummy_cfg = BuildingConfig()
    flux_results = {}
    
    for face, az in orientations.items():
        strat = NoShading(1.5, 2.0, az)
        sim = ThermalSystem(dummy_cfg, strat, LAT, LON)
        res = sim.simulate(weather_df.copy())
        
        # Now shift time to Local for Visualization
        res['time'] = res['time'] + pd.Timedelta(hours=TZ_OFFSET)
        flux_results[face] = res[['time', 'T_out', 'Solar_Flux']] 

    # Plot Heatmaps
    fig_hm, axes_hm = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    
    # Find global max
    gmax = 0
    for f in orientations:
        gmax = max(gmax, flux_results[f]['Solar_Flux'].max())
    
    for ax, face in zip(axes_hm, ['North', 'East', 'South', 'West']):
        df_res = flux_results[face]
        pivot_data = df_res.copy()
        pivot_data['Day'] = pivot_data['time'].dt.dayofyear
        pivot_data['Hour'] = pivot_data['time'].dt.hour
        
        grid = pivot_data.groupby(['Day', 'Hour'])['Solar_Flux'].mean().unstack()
        # Reindex to ensure 0-23 hours
        grid = grid.reindex(columns=range(24), fill_value=0)
        
        im = ax.imshow(grid, aspect='auto', cmap='inferno', vmin=0, vmax=gmax, origin='lower', extent=[0, 24, 1, 365])
        ax.set_title(f'{face} Façade', fontsize=14)
        ax.set_xlabel('Hour of Day')
        if face == 'North':
            ax.set_ylabel('Day of Year')
    
    cbar = fig_hm.colorbar(im, ax=axes_hm.ravel().tolist(), pad=0.01)
    cbar.set_label('Incident Irradiance ($W/m^2$)', rotation=270, labelpad=15)
    plt.suptitle('Annual Solar Heat Gain Distribution by Orientation (Sungrove)', fontsize=16, y=1.05)
    plt.savefig(os.path.join(output_dir, '1_Solar_Heatmaps_4Faces.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 2. Three Strategies Comparison (Single vs Combined)
    # ---------------------------------------------------------
    print("Generating 2. Strategy Comparison Curves...")
    
    CRITICAL_AZIMUTH = 90 # West

    # Base Config
    cfg_base = BuildingConfig(
        Q_internal=500.0, k_wall=0.8, c_wall=1000.0, rho_wall=1800.0,
        u_window=5.8, tau_window=0.85, 
        night_cooling=False
    )
    strat_base = NoShading(2.0, 2.0, CRITICAL_AZIMUTH)
    
    strat_shade = OverhangStats(2.0, 2.0, CRITICAL_AZIMUTH, depth_L=1.5)
    
    cfg_glass = BuildingConfig(
        Q_internal=500.0, k_wall=0.8, c_wall=1000.0, rho_wall=1800.0,
        u_window=1.8, tau_window=0.4, 
        night_cooling=False
    )
    
    cfg_vent = BuildingConfig(
        Q_internal=500.0, k_wall=0.8, c_wall=1000.0, rho_wall=1800.0,
        u_window=5.8, tau_window=0.85,
        night_cooling=True, night_vent_rate=8.0
    )
    
    cfg_comb = BuildingConfig(
        Q_internal=500.0, k_wall=0.8, c_wall=1000.0, rho_wall=1800.0,
        u_window=1.8, tau_window=0.4,
        night_cooling=True, night_vent_rate=8.0
    )
    strat_comb = OverhangStats(2.0, 2.0, CRITICAL_AZIMUTH, depth_L=1.5)

    print("Running simulations for Day 130 period...")
    
    sim_base = ThermalSystem(cfg_base, strat_base, LAT, LON)
    sim_shade = ThermalSystem(cfg_base, strat_shade, LAT, LON)
    sim_glass = ThermalSystem(cfg_glass, strat_base, LAT, LON)
    sim_vent = ThermalSystem(cfg_vent, strat_base, LAT, LON)
    sim_comb = ThermalSystem(cfg_comb, strat_comb, LAT, LON)
    
    # Simulate with UTC
    res_base = sim_base.simulate(weather_df.copy())
    res_shade = sim_shade.simulate(weather_df.copy())
    res_glass = sim_glass.simulate(weather_df.copy())
    res_vent = sim_vent.simulate(weather_df.copy())
    res_comb = sim_comb.simulate(weather_df.copy())
    
    # SHIFT Time to Local for Analysis
    for r in [res_base, res_shade, res_glass, res_vent, res_comb]:
        r['time'] = r['time'] + pd.Timedelta(hours=TZ_OFFSET)
    
    # Filter 72h (3 Days)
    start_day = 130
    end_day = 133
    mask = (res_base['time'].dt.dayofyear >= start_day) & (res_base['time'].dt.dayofyear < end_day)
    
    t_axis = res_base.loc[mask, 'time']
    
    # Auto-detect Unit
    T_mean_chk = res_base.loc[mask, 'T_in'].mean()
    if T_mean_chk > 200:
        offset = 273.15
    else:
        offset = 0.0

    # Convert Temps
    T_base_C = res_base.loc[mask, 'T_in'] - offset
    T_shade_C = res_shade.loc[mask, 'T_in'] - offset
    T_glass_C = res_glass.loc[mask, 'T_in'] - offset
    T_vent_C = res_vent.loc[mask, 'T_in'] - offset
    T_comb_C = res_comb.loc[mask, 'T_in'] - offset
    T_out_C = res_base.loc[mask, 'T_out'] 

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(t_axis, T_base_C, label='Baseline (Unoptimized)', color='k', linestyle='-', linewidth=2.5, zorder=10)
    plt.plot(t_axis, T_shade_C, label='+ Shading', color=c_shade, linewidth=2)
    plt.plot(t_axis, T_glass_C, label='+ Low-E Glass', color='#8e44ad', linewidth=2)
    plt.plot(t_axis, T_vent_C, label='+ Night Vent', color=c_vent, linewidth=2)
    plt.plot(t_axis, T_comb_C, label='Combined Solution', color=c_comb, linewidth=3)
    plt.plot(t_axis, T_out_C, label='Outdoor Temp', color='gray', linestyle='--', linewidth=2, alpha=0.8)
    
    plt.title(f'Temperature Response: Passive Cooling Strategies (Days {start_day}-{end_day-1})', fontsize=14)
    plt.ylabel('Indoor Temperature (°C)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    plt.legend(loc='upper right', ncol=2)
    # Fix Y-lim for visual clarity
    plt.ylim(25, 50) 
    plt.savefig(os.path.join(output_dir, '2_Strategy_Comparison_Curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 3. Combined Effects Dashboard
    # ---------------------------------------------------------
    print("Generating 3. Combined Effects Dashboard (3 Days)...")
    
    day_res_base = res_base.loc[mask].copy()
    day_res_comb = res_comb.loc[mask].copy()
    t_chunk = day_res_base['time']

    # Calc Illuminance
    tau_vis = 0.65
    efficacy = 105.0 
    c_room = 0.5 
    win_to_floor = 0.175
    
    flux_comb = day_res_comb['Solar_Flux'].clip(lower=0)
    shade_factor = day_res_comb['F_shade']
    
    # 10% diffuse transmission floor
    effective_transmission = (1 - shade_factor).clip(lower=0.1) 
    
    illum_comb = flux_comb * efficacy * tau_vis * effective_transmission * c_room * win_to_floor
    
    flux_base = day_res_base['Solar_Flux'].clip(lower=0)
    illum_base = flux_base * efficacy * 0.85 * 1.0 * c_room * win_to_floor

    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_chunk, T_base_C, label='Baseline', color=c_comb, linewidth=2)
    ax1.plot(t_chunk, T_comb_C, label='Retrofit', color=c_shade, linewidth=2)
    ax1.plot(t_chunk, T_out_C, label='Outdoor', color='gray', linestyle='--', alpha=0.7)
    ax1.fill_between(t_chunk, 20, 26, color='#2ecc71', alpha=0.15, label='Comfort Zone')
    ax1.set_title('Free-Running Temperature', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Temp (°C)')
    ax1.legend(loc='upper right', ncol=3)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    load_base = day_res_base['Q_cooling_load'] / 1000.0
    load_comb = day_res_comb['Q_cooling_load'] / 1000.0
    ax2.fill_between(t_chunk.values, 0, load_base.values, color=c_comb, alpha=0.3, label='Baseline Load')
    ax2.fill_between(t_chunk.values, 0, load_comb.values, color=c_shade, alpha=0.6, label='Retrofit Load')
    ax2.set_title('Cooling Load Profile', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Power (kW)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t_chunk, illum_base, label='Baseline', color='gray', linestyle=':', alpha=0.5)
    ax3.plot(t_chunk, illum_comb, label='Optimized', color='#d4ac0d', linewidth=2.5)
    ax3.axhline(2000, color='red', linestyle='--', label='Glare Threshold')
    ax3.axhline(300, color='green', linestyle='--', label='Min Req')
    ax3.set_title('Visual Comfort', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Lux')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_Combined_Effects_Dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 3b. Export Night Ventilation Data (CSV)
    # ---------------------------------------------------------
    print("Generating 3b. Night Ventilation Data CSV...")
    
    # Run a short simulation for Summer Solstice (Day 172)
    start_day_sol = 172
    
    # Use Combined Strategy (Retrofit) which has night cooling enabled
    sim_night = ThermalSystem(cfg_comb, strat_comb, LAT, LON)
    res_night = sim_night.simulate(weather_df.copy())
    
    # Shift Time
    res_night['time'] = res_night['time'] + pd.Timedelta(hours=TZ_OFFSET)
    
    # Filter for single day
    mask_sol = (res_night['time'].dt.dayofyear == start_day_sol)
    day_data = res_night.loc[mask_sol].copy()
    
    # Resample to hourly
    day_hourly = day_data.set_index('time').resample('H').mean(numeric_only=True)
    day_hourly['Hour'] = day_hourly.index.hour
    
    # Extract Ventilation Cooling (Q_vent_loss < 0 means cooling)
    # Convert to Watts or kW? User asked for data table. W is fine, or kW.
    # Q_vent_loss is in Watts (J/s) roughly from calculate.py logic (m_dot_cp * dT)
    # Let's use Watts for precision or kW for readability. Let's use kW.
    
    # Q_vent_loss: positive = gain (hot air in), negative = loss (cooling)
    # We want "Cooling Rate" (positive value)
    day_hourly['Ventilation_Cooling_kW'] = -day_hourly['Q_vent_loss'].clip(upper=0) / 1000.0
    day_hourly['Indoor_Temp_C'] = day_hourly['T_in']
    day_hourly['Outdoor_Temp_C'] = day_hourly['T_out']
    
    # Select relevant columns
    csv_cols = ['Hour', 'Indoor_Temp_C', 'Outdoor_Temp_C', 'Ventilation_Cooling_kW']
    csv_path = os.path.join(csv_dir, 'sungrove_night_ventilation_cooling.csv')
    day_hourly[csv_cols].to_csv(csv_path, index=False, float_format='%.2f')
    print(f"Exported detailed night ventilation data to {csv_path}")

    # Generate Summary Table for Paper
    # Group by Period: Night (20:00-06:59) vs Day (07:00-19:59)
    # Night cooling active hours: 20,21,22,23 and 0,1,2,3,4,5,6 (11 hours)
    is_night = (day_hourly.index.hour < 7) | (day_hourly.index.hour >= 20)
    day_hourly['Period'] = np.where(is_night, 'Night Cooling (Active)', 'Daytime (Passive)')

    summary = day_hourly.groupby('Period').agg({
        'Indoor_Temp_C': 'mean',
        'Outdoor_Temp_C': 'mean',
        'Ventilation_Cooling_kW': ['mean', 'max', 'sum'] # Sum works as kWh because index is hourly
    }).round(2)
    
    # Clean up column names for final CSV
    summary.columns = ['Avg_Indoor_Temp_C', 'Avg_Outdoor_Temp_C', 'Avg_Cooling_Rate_kW', 'Peak_Cooling_Rate_kW', 'Total_Cooling_Energy_kWh']
    
    summary_path = os.path.join(csv_dir, 'sungrove_night_ventilation_summary.csv')
    summary.to_csv(summary_path)
    print(f"Exported summary table to {summary_path}")

    # ---------------------------------------------------------
    # 4. Seasonal Temperature Comparison
    # ---------------------------------------------------------
    print("Generating 4. Seasonal Temperature Comparison...")
    
    season_days = {
        'Spring (Mar 21)': 80,
        'Summer (Jun 21)': 172,
        'Autumn (Sep 23)': 266,
        'Winter (Dec 21)': 355
    }
    
    fig_sea, axes_sea = plt.subplots(2, 2, figsize=(15, 10))
    axes_sea = axes_sea.ravel()
    
    for i, (season_name, day_idx) in enumerate(season_days.items()):
        ax = axes_sea[i]
        
        mask_s = (res_base['time'].dt.dayofyear == day_idx)
        if not mask_s.any(): continue 
        
        t_s = res_base.loc[mask_s, 'time']
        base_s = res_base.loc[mask_s, 'T_in'] - offset
        comb_s = res_comb.loc[mask_s, 'T_in'] - offset
        out_s = res_base.loc[mask_s, 'T_out']
        
        # FIX: Explicitly plot variables to ensure they appear
        ax.plot(t_s, base_s, label='Baseline', color='k', linewidth=2)
        ax.plot(t_s, comb_s, label='Optimized', color=c_shade, linewidth=2)
        ax.plot(t_s, out_s, label='Outdoor', color='gray', linestyle='--')
        
        ax.axhspan(20, 26, color='green', alpha=0.1)
        ax.set_title(season_name, fontsize=12)
        ax.set_ylabel('Temp (°C)')
        
        # Dynamic Y-limits per subplot to minimize whitespace and avoid clipping
        # Calculate min/max across all plotted series
        data_min = min(base_s.min(), comb_s.min(), out_s.min())
        data_max = max(base_s.max(), comb_s.max(), out_s.max())
        
        # Ensure Comfort Zone (20-26) is somewhat visible if close, 
        # but don't force it if data is far away (like 45C)
        
        margin = 2.0 # degrees padding
        ax.set_ylim(data_min - margin, data_max + margin)
        
        if i == 0: ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(True, linestyle=':')

    plt.suptitle('Seasonal Performance Comparison (Sungrove)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_Seasonal_Temperature_Comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Sungrove analysis complete.")


def run_borealis_part2():
    print("\n=== Running Part 2: Aurora/Borealis (Student Union) ===")
    
    # Paths
    weather_file = os.path.join(current_dir, '..', '..', 'data', 'weather', 'norway_data.csv')
    output_dir = os.path.join(current_dir, '..', '..', 'data', 'model_images', 'student_union')
    csv_dir = os.path.join(current_dir, '..', '..', 'data', 'csv', 'student_union')
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(csv_dir): os.makedirs(csv_dir)

    # Norway UTC+1
    TZ_OFFSET = 1
    weather_df = load_weather_robust(weather_file, timezone_offset=TZ_OFFSET, apply_offset_to_cols=False)
    if weather_df.empty:
        print("Error: Norway weather data missing.")
        return

    LAT = 60.5
    LON = 10.7 # Oslo approx

    # ---------------------------------------------------------
    # 1. Thermal Mass Effect (Temperature Amplitude)
    # ---------------------------------------------------------
    print("Generating B1. Thermal Mass Comparison...")
    
    # 3 Scenarios: Light, Medium (Baseline), Heavy
    
    # Light Mass (Wood Frame, High Insulation)
    cfg_light = BuildingConfig(
        k_wall=0.15, rho_wall=600.0, c_wall=1200.0, layer_thickness=0.2, 
        u_window=1.2,
        ventilation_rate=2.0 
    )
    
    # Medium Mass (Brick/Standard)
    cfg_med = BuildingConfig(
        k_wall=0.6, rho_wall=1600.0, c_wall=840.0, layer_thickness=0.3, # Brick
        u_window=1.2,
        ventilation_rate=2.0
    )
    
    # Heavy Mass (Concrete/Stone)
    cfg_heavy = BuildingConfig(
        k_wall=1.8, rho_wall=2400.0, c_wall=1000.0, layer_thickness=0.5, # Thick Stone
        u_window=1.2,
        ventilation_rate=2.0
    )
    
    # Strategy: South facing, no shading
    strat = NoShading(2.0, 2.0, 0)
    
    # Manual C_in override
    sim_light = ThermalSystem(cfg_light, strat, LAT, LON)
    sim_light.C_in = 200000.0 
    
    sim_med = ThermalSystem(cfg_med, strat, LAT, LON)
    sim_med.C_in = 800000.0 # Baseline
    
    sim_heavy = ThermalSystem(cfg_heavy, strat, LAT, LON)
    sim_heavy.C_in = 4000000.0 # High mass
    
    res_light = sim_light.simulate(weather_df.copy())
    res_med = sim_med.simulate(weather_df.copy())
    res_heavy = sim_heavy.simulate(weather_df.copy())
    
    # Apply Timezone Offset for Plotting
    for r in [res_light, res_med, res_heavy]:
        r['time'] = r['time'] + pd.Timedelta(hours=TZ_OFFSET)
    
    # Pick a period: Early Spring (Cold outside, Strong Sun)
    # March 10 - March 24 (Day 69 - 83)
    start_d, end_d = 69, 83
    mask = (res_light['time'].dt.dayofyear >= start_d) & (res_light['time'].dt.dayofyear <= end_d)
    
    t_chunk = res_light.loc[mask, 'time']
    
    # Unit Chk
    t_mean = res_med.loc[mask, 'T_in'].mean()
    off = 273.15 if t_mean > 200 else 0
    convert = lambda x: x - off

    plt.figure(figsize=(10, 5))
    plt.plot(t_chunk, convert(res_light.loc[mask, 'T_in']), label='Lightweight Mass', color='#e67e22', linestyle='--')
    plt.plot(t_chunk, convert(res_med.loc[mask, 'T_in']), label='Baseline (Medium Mass)', color='gray', linestyle='-')
    plt.plot(t_chunk, convert(res_heavy.loc[mask, 'T_in']), label='Heavy Thermal Mass', color='#8e44ad', linewidth=2.5)
    
    # Outdoor
    plt.plot(t_chunk, res_light.loc[mask, 'T_out'], label='Outdoor Temp', color='#3498db', linestyle=':', alpha=0.6)
    
    # Calculate Amplitude reduction (avg daily swing)
    # Just show max swing in legend
    amp_light = convert(res_light.loc[mask, 'T_in']).max() - convert(res_light.loc[mask, 'T_in']).min()
    amp_heavy = convert(res_heavy.loc[mask, 'T_in']).max() - convert(res_heavy.loc[mask, 'T_in']).min()
    
    plt.title(f'Thermal Mass Effect: Damping Temperature Swings (March)\nMax Swing: Light={amp_light:.1f}K vs Heavy={amp_heavy:.1f}K', fontsize=12)
    plt.ylabel('Indoor Temperature (°C)')
    plt.ylim(-5, 35) 
    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(output_dir, '1_ThermalMass_Effect.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save CSV
    pd.DataFrame({
        'Time': t_chunk.values,
        'T_Light': convert(res_light.loc[mask, 'T_in']).values,
        'T_Medium': convert(res_med.loc[mask, 'T_in']).values,
        'T_Heavy': convert(res_heavy.loc[mask, 'T_in']).values,
        'T_Outdoor': res_light.loc[mask, 'T_out'].values
    }).to_csv(os.path.join(csv_dir, 'thermal_mass_compare.csv'), index=False)

    # ---------------------------------------------------------
    # B4. Seasonal Comparison (Borealis)
    # ---------------------------------------------------------
    print("Generating B4. Seasonal Comparison (Borealis)...")
    
    season_days = {
        'Spring (Mar 21)': 80,
        'Summer (Jun 21)': 172,
        'Autumn (Sep 23)': 266,
        'Winter (Dec 21)': 355
    }
    
    fig_sea, axes_sea = plt.subplots(2, 2, figsize=(15, 10))
    axes_sea = axes_sea.ravel()
    
    for i, (season_name, day_idx) in enumerate(season_days.items()):
        ax = axes_sea[i]
        
        mask_s = (res_med['time'].dt.dayofyear == day_idx)
        if not mask_s.any(): continue
        
        t_s = res_med.loc[mask_s, 'time']
        med_s = convert(res_med.loc[mask_s, 'T_in'])
        heavy_s = convert(res_heavy.loc[mask_s, 'T_in'])
        out_s = res_med.loc[mask_s, 'T_out']
        
        ax.plot(t_s, med_s, label='Baseline', color='gray', linestyle='--')
        ax.plot(t_s, heavy_s, label='Optimized', color='#8e44ad', linewidth=2.5)
        ax.plot(t_s, out_s, label='Outdoor', color='#3498db', linestyle=':', alpha=0.7)
        
        ax.axhspan(18, 24, color='#f1c40f', alpha=0.1, label='Comfort Zone')
        
        ax.set_title(season_name)
        ax.set_ylabel('Temp (°C)')
        
        # Dynamic Y-limits per subplot to minimize whitespace
        data_min = min(med_s.min(), heavy_s.min(), out_s.min())
        data_max = max(med_s.max(), heavy_s.max(), out_s.max())
        margin = 2.0
        ax.set_ylim(data_min - margin, data_max + margin)
        
        if i == 0: ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(True, linestyle=':')

    plt.suptitle('Seasonal Performance: Thermal Mass Damping (Borealis)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_Seasonal_Temperature_Comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 2. Building Geometry/Material/Glass (Heat Loss Factors)
    # ---------------------------------------------------------
    print("Generating B2. Passive Design Factors...")
    
    # Same logic - use UTC weather for simulation
    cfg_b2_base = BuildingConfig(k_wall=0.5, u_window=2.5)
    strat_b2_base = NoShading(2.0, 2.0, 180) # North
    sim_base = ThermalSystem(cfg_b2_base, strat_b2_base, LAT, LON)
    # Simulator handles UTC correctly? Yes if solar geom is correct.
    load_base = sim_base.simulate(weather_df.copy())['Q_heating_load'].sum() / 1e6
    
    strat_b2_ori = NoShading(2.0, 2.0, 0) # South
    sim_ori = ThermalSystem(cfg_b2_base, strat_b2_ori, LAT, LON)
    load_ori = sim_ori.simulate(weather_df.copy())['Q_heating_load'].sum() / 1e6
    
    cfg_b2_ins = BuildingConfig(k_wall=0.04, u_window=2.5) 
    sim_ins = ThermalSystem(cfg_b2_ins, strat_b2_ori, LAT, LON)
    load_ins = sim_ins.simulate(weather_df.copy())['Q_heating_load'].sum() / 1e6
    
    cfg_b2_glass = BuildingConfig(k_wall=0.04, u_window=0.8)
    sim_glass = ThermalSystem(cfg_b2_glass, strat_b2_ori, LAT, LON)
    load_glass = sim_glass.simulate(weather_df.copy())['Q_heating_load'].sum() / 1e6

    labels = ['Baseline\n(North)', '+ Orientation\n(South)', '+ Material\n(Insulation)', '+ Glass\n(Triple Glazing)']
    values = [load_base, load_ori, load_ins, load_glass]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['#95a5a6', '#f39c12', '#27ae60', '#2980b9'])
    plt.plot(labels, values, color='black', marker='o', linestyle='--', alpha=0.3)
    
    plt.ylabel('Annual Heating Energy (MJ)')
    plt.title('Impact of Passive Design Strategies in Aurora', fontsize=14)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
    plt.savefig(os.path.join(output_dir, '2_Passive_Design_Factors.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 3. Annual Energy
    # ---------------------------------------------------------
    print("Generating B3. Annual Energy Profile...")
    
    best_res = sim_glass.simulate(weather_df.copy())
    # Shift Time for month grouping (important for boundary cases)
    best_res['time'] = best_res['time'] + pd.Timedelta(hours=TZ_OFFSET)
    
    best_res['Month'] = best_res['time'].dt.month
    monthly = best_res.groupby('Month')[['Q_heating_load', 'Q_cooling_load', 'Q_sol_gain']].sum() / 1e6 
    
    plt.figure(figsize=(10, 6))
    plt.bar(monthly.index, monthly['Q_heating_load'], label='Heating Load', color='#e74c3c')
    plt.bar(monthly.index, monthly['Q_cooling_load'], bottom=monthly['Q_heating_load'], label='Cooling Load', color='#3498db')
    
    ax2 = plt.gca().twinx()
    ax2.plot(monthly.index, monthly['Q_sol_gain'], color='#f1c40f', marker='o', linewidth=2, label='Solar Gain Potential')
    ax2.set_ylabel('Passive Solar Gain (MJ)')
    
    plt.xlabel('Month')
    plt.ylabel('HVAC Load (MJ)')
    plt.title('Optimized Annual Energy Profile (Borealis)', fontsize=14)
    plt.xticks(range(1, 13))
    
    h1, l1 = plt.gca().get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1+h2, l1+l2, loc='upper center')
    
    plt.savefig(os.path.join(output_dir, 'B2_Annual_Energy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    monthly.to_csv(os.path.join(csv_dir, 'annual_energy_profile.csv'))
    
    print("Borealis analysis complete.")

if __name__ == "__main__":
    run_sungrove_part1()
    run_borealis_part2()
