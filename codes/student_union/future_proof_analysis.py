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

def run_future_proof_design():
    print("\n--- Running Future-Proof Design Analysis for Sungrove Student Union ---")
    output_dir = os.path.join(current_dir, '..', '..', 'data', 'model_images', 'student_union')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 1. Base Weather (Current)
    base_file = os.path.join(current_dir, '..', '..', 'data', 'weather', 'singapore_data.csv')
    df_current = load_weather_robust(base_file)
    if df_current.empty: return

    # 2. Future Weather (2050 Projection)
    # Assumption for Tropics: +2.0°C mean temp rise, +10% Radiation (less cloud/aerosol change?)
    # More extreme heat waves.
    df_future = df_current.copy()
    df_future['T2m'] = df_future['T2m'] + 2.0
    
    # 3. Design Configuration: "The Climate Membrane"
    # Adaptive shading + Night Vent + High Mass (Hybrid)
    
    # Strategy A: Current Code (Static Shading)
    cfg_static = BuildingConfig(
        C_in=800000.0, Q_internal=1000.0, # High load for Student Union (people + computers)
        layer_thickness=0.3, wall_area=1.0, window_ratio=0.5, # Open views
        k_wall=0.8, rho_wall=1800.0, c_wall=1000.0, 
        h_in=8.0, h_out=20.0,
        u_window=2.0, tau_window=0.6, k_const_absorb=0.5,
        ventilation_rate=2.0, room_volume=50.0 # Standard vent
    )
    strat_static = OverhangStats(2.0, 2.0, 0, 1.0) # L=1.0m Overhang

    # Strategy B: Future-Proof Adaptive (Dynamic)
    # Higher Mass to buffer heat waves
    # Better Glass
    # Smart Vent
    cfg_adaptive = BuildingConfig(
        C_in=1200000.0, # Increased Thermal Mass
        Q_internal=1000.0, 
        layer_thickness=0.4, wall_area=1.0, window_ratio=0.5, 
        k_wall=0.8, rho_wall=2000.0, c_wall=1000.0, 
        h_in=8.0, h_out=20.0,
        u_window=1.2, tau_window=0.35, k_const_absorb=0.3, # Low-E, Cool Roof
        ventilation_rate=2.0, room_volume=50.0,
        night_cooling=True, night_vent_rate=10.0 # Aggressive Night Flush
    )
    # Adaptive shading implied by lower tau_window (e.g. electrochromic) or deeper static
    strat_adaptive = OverhangStats(2.0, 2.0, 0, 2.0) # Deeper Overhang L=2.0m

    LAT, LON = 1.35, 103.8

    # Run Simulations
    print("Simulating Current Climate...")
    sim_stat_curr = ThermalSystem(cfg_static, strat_static, LAT, LON)
    res_stat_curr = sim_stat_curr.simulate(df_current)

    sim_adap_curr = ThermalSystem(cfg_adaptive, strat_adaptive, LAT, LON)
    res_adap_curr = sim_adap_curr.simulate(df_current)

    print("Simulating Future Climate (2050)...")
    sim_stat_fut = ThermalSystem(cfg_static, strat_static, LAT, LON)
    res_stat_fut = sim_stat_fut.simulate(df_future)

    sim_adap_fut = ThermalSystem(cfg_adaptive, strat_adaptive, LAT, LON)
    res_adap_fut = sim_adap_fut.simulate(df_future)

    # --- Analysis 1: Resilience (Temperature Exceedance) ---
    # Count hours > 28°C (Discomfort)
    # Note in Sungrove (Tropics), T_out is often > 28.
    # Without Cooling, T_in will be > 28 most of the time.
    # We should look at "Degree Hours > 28" or "Hours > 30" to see severity.
    # Or check Cooling Load if AC is on.
    # Let's count "Hours where T_in > 29" for passive drift scenario?
    # Actually, the simulation has AC setpoint at 26. So T_in is clamped at ~26 if Q_cooling > 0.
    # If we want to test PASSIVE resilience, we should turn off AC calculation or look at Load.
    # But Q_cooling_load IS the metric for "Performance".
    # Let's use Cooling Load for the bar chart instead of Discomfort Hours, as Discomfort is 0 by definition if AC works.
    # OR: Define "Passive Discomfort" as if AC failed. But current code clamps T_in for load calc.
    # Let's stick to Cooling Load (Energy) and Peak Load.

    # Re-purpose Chart 1: Seasonal Peak Load Comparison
    peak_stat_curr = res_stat_curr['Q_cooling_load'].max()
    peak_stat_fut = res_stat_fut['Q_cooling_load'].max()
    peak_adap_curr = res_adap_curr['Q_cooling_load'].max()
    peak_adap_fut = res_adap_fut['Q_cooling_load'].max()

    labels = ['Current Climate', 'Future Climate (2050)']
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, [peak_stat_curr, peak_stat_fut], width, label='Strategy A: Static Code', color='#95a5a6')
    plt.bar(x + width/2, [peak_adap_curr, peak_adap_fut], width, label='Strategy B: Future-Proof Adaptive', color='#2ecc71')
    
    plt.ylabel('Peak Cooling Power (W)')
    plt.title('Student Union Resilience: Peak Demand 2050')
    plt.xticks(x, labels)
    plt.legend()
    
    # Annotate percent reduction
    red_curr = 100 * (peak_stat_curr - peak_adap_curr) / peak_stat_curr
    red_fut = 100 * (peak_stat_fut - peak_adap_fut) / peak_stat_fut
    
    plt.text(x[0], peak_stat_curr, f'-{red_curr:.0f}%', ha='center', va='bottom', fontweight='bold')
    plt.text(x[1], peak_stat_fut, f'-{red_fut:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.savefig(os.path.join(output_dir, 'future_proof_peak_load.png'))
    plt.close()

    # --- Analysis 2: Cooling Load (Energy) ---
    load_stat_curr = res_stat_curr['Q_cooling_load'].sum() / 1e6
    load_stat_fut = res_stat_fut['Q_cooling_load'].sum() / 1e6
    load_adap_curr = res_adap_curr['Q_cooling_load'].sum() / 1e6
    load_adap_fut = res_adap_fut['Q_cooling_load'].sum() / 1e6
    
    plt.figure(figsize=(8, 6))
    plt.plot(['Current', 'Future'], [load_stat_curr, load_stat_fut], marker='o', label='Static Code', linestyle='--', color='#e74c3c')
    plt.plot(['Current', 'Future'], [load_adap_curr, load_adap_fut], marker='o', label='Adaptive Design', linewidth=3, color='#27ae60')
    
    plt.ylabel('Annual Cooling Load (Mega-Units)')
    plt.title('Energy Sensitivity to Climate Change')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'future_proof_energy.png'))
    plt.close()
    
    # --- Analysis 3: Daylighting vs Glare Tradeoff (Proxy) ---
    # Compare "Solar Flux Entering" vs "Shade Factor"
    # Adaptive has LOW Tau (0.35) but DEEP Shade.
    # We want to see if we killed daylight.
    # Proxy: Assume "Useful Daylight" is Flux between 100 and 2000 W (very rough).
    # Just plot Mean Daily Flux Entering Window.
    
    mean_flux_stat = res_stat_curr['Solar_Flux'].mean() * cfg_static.tau_window * (1 - res_stat_curr['F_shade'].mean())
    mean_flux_adap = res_adap_curr['Solar_Flux'].mean() * cfg_adaptive.tau_window * (1 - res_adap_curr['F_shade'].mean())
    
    print(f"Mean Solar Flux Entering (Static): {mean_flux_stat:.1f} W/m2")
    print(f"Mean Solar Flux Entering (Adaptive): {mean_flux_adap:.1f} W/m2")
    print("Note: Adaptive drastically reduces heat, but may require artificial light planning if too dark.")

if __name__ == "__main__":
    run_future_proof_design()
