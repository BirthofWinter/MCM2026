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

def load_weather_simple(filepath):
    # Simplified loader for quick estimation
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
        for c in ['Gb(i)', 'T2m']:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except:
        return pd.DataFrame()

def run_strategic_roadmap_fitting():
    print("\n--- Running Strategic Roadmap Fitting for Sungrove University ---")
    weather_file = os.path.join(current_dir, '..', '..', 'data', 'weather', 'singapore_data.csv')
    df_weather = load_weather_simple(weather_file)
    if df_weather.empty: return

    LAT, LON = 1.35, 103.8
    # Define phased intervention packages with increasing cost and complexity
    
    # Base Config (Current State)
    cfg_base = BuildingConfig(
        C_in=800000.0, Q_internal=500.0, layer_thickness=0.3, wall_area=10.0, window_ratio=0.45,
        k_wall=0.8, rho_wall=1800.0, c_wall=1000.0, h_in=8.0, h_out=20.0,
        u_window=2.5, tau_window=0.7, k_const_absorb=0.6,
        ventilation_rate=2.0, room_volume=40.0, night_cooling=False
    )
    strat_base = NoShading(2.0, 2.0, 0) # South Facing

    # Phase 1: "Low Hanging Fruit" - Static Shading + White Paint
    # Low Cost, High Impact
    cfg_p1 = BuildingConfig(
        C_in=800000.0, Q_internal=500.0, layer_thickness=0.3, wall_area=10.0, window_ratio=0.45,
        k_wall=0.8, rho_wall=1800.0, c_wall=1000.0, h_in=8.0, h_out=20.0,
        u_window=2.5, tau_window=0.7, 
        k_const_absorb=0.35, # White Paint (Cheap)
        ventilation_rate=2.0, room_volume=40.0, night_cooling=False
    )
    strat_p1 = OverhangStats(2.0, 2.0, 0, 1.2) # Simple 1.2m Overhang

    # Phase 2: "Deep Retrofit" - Smart Glass + Night Vent
    # High Cost, Max Performance
    cfg_p2 = BuildingConfig(
        C_in=800000.0, Q_internal=500.0, layer_thickness=0.3, wall_area=10.0, window_ratio=0.45,
        k_wall=0.8, rho_wall=1800.0, c_wall=1000.0, h_in=8.0, h_out=20.0,
        u_window=1.5, tau_window=0.35, # Low-E / Smart Glass (Expensive)
        k_const_absorb=0.30, 
        ventilation_rate=2.0, room_volume=40.0, 
        night_cooling=True, night_vent_rate=8.0 # Automation Control
    )
    strat_p2 = OverhangStats(2.0, 2.0, 0, 1.5) # Deeper Overhang

    # Simulation
    def get_annual_load(cfg, strat):
        sim = ThermalSystem(cfg, strat, LAT, LON)
        res = sim.simulate(df_weather)
        return res['Q_cooling_load'].sum() / 1e6 # MJ

    print("Simulating Baseline...")
    load_base = get_annual_load(cfg_base, strat_base)
    print("Simulating Phase 1...")
    load_p1 = get_annual_load(cfg_p1, strat_p1)
    print("Simulating Phase 2...")
    load_p2 = get_annual_load(cfg_p2, strat_p2)

    # ROI Calculation (Hypothetical Cost Units CU)
    # Energy Cost: 0.2 CU / MJ
    cost_energy = 0.2
    
    # Investment Costs
    # Phase 1: Paint (100) + Aluminum Fins (300) = 400
    # Phase 2: Glass (1000) + Sensors (200) + Phase 1 = 1600
    inv_p1 = 400
    inv_p2 = 1600
    
    save_p1 = (load_base - load_p1) * cost_energy
    save_p2 = (load_base - load_p2) * cost_energy
    
    roi_p1 = save_p1 / inv_p1
    roi_p2 = save_p2 / inv_p2
    
    payback_p1 = inv_p1 / save_p1
    payback_p2 = inv_p2 / save_p2

    print(f"\n--- Strategic Analysis Results ---")
    print(f"Baseline Load: {load_base:.1f} MJ")
    print(f"Phase 1 Load: {load_p1:.1f} MJ (-{100*(load_base-load_p1)/load_base:.1f}%) | Payback: {payback_p1:.1f} yrs")
    print(f"Phase 2 Load: {load_p2:.1f} MJ (-{100*(load_base-load_p2)/load_base:.1f}%) | Payback: {payback_p2:.1f} yrs")

    # Plot Pareto Chart
    output_dir = os.path.join(current_dir, '..', '..', 'data', 'model_images', 'optimization_letter')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    plt.figure(figsize=(8, 6))
    
    investments = [0, inv_p1, inv_p2]
    savings = [0, load_base - load_p1, load_base - load_p2]
    labels = ['Baseline', 'Phase 1: Quick Wins\n(Shading+Paint)', 'Phase 2: Deep Retrofit\n(Glass+Smart Vent)']
    
    plt.plot(investments, savings, marker='o', linestyle='-', linewidth=2, color='#2980b9')
    
    for i, txt in enumerate(labels):
        plt.annotate(txt, (investments[i], savings[i]), xytext=(0, 10), textcoords='offset points', ha='center')
        
    plt.xlabel('Estimated Investment Cost (Hypothetical Units)')
    plt.ylabel('Annual Energy Savings (MJ)')
    plt.title('Cost-Benefit Analysis of Retrofit Strategies')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'strategic_roadmap_pareto.png'))
    plt.close()

if __name__ == "__main__":
    run_strategic_roadmap_fitting()
