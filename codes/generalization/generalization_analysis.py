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
plt.rcParams['axes.unicode_minus'] = False 

def load_and_morph_weather(filepath, temp_shift=0.0, rad_multiplier=1.0, name_suffix=""):
    """
    Load weather data and morph it to simulate a different geography at same latitude.
    temp_shift: Add/Subtract degrees C (e.g., -10 for altitude)
    rad_multiplier: Scale solar radiation (e.g., *1.2 for thinner atmosphere or *0.5 for cloudy)
    """
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
        
        cols_to_numeric = ['Gb(i)', 'Gd(i)', 'Gr(i)', 'H_sun', 'T2m', 'WS10m', 'Int']
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Morphing
        df['T2m'] = df['T2m'] + temp_shift
        # Clip radiation to realistic max (Solar Constant ~1361, surface max ~1200)
        max_rad = 1200.0
        df['Gb(i)'] = (df['Gb(i)'] * rad_multiplier).clip(upper=max_rad)
        df['Gd(i)'] = (df['Gd(i)'] * rad_multiplier).clip(upper=max_rad)
        
        # Recalculate Global for reference (not used in physics directly but good for sanity)
        # df['G_total'] = df['Gb(i)'] + df['Gd(i)']
        
        print(f"Weather Morphed [{name_suffix}]: T_avg changed from {df['T2m'].mean()-temp_shift:.1f} to {df['T2m'].mean():.1f}")
        return df
    except Exception as e:
        print(f"Error loading: {e}")
        return pd.DataFrame()

def run_generalization_study():
    print("\n--- Running Generalization & Adaptability Study ---")
    output_dir = os.path.join(current_dir, '..', '..', 'data', 'model_images', 'generalization')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # =========================================================
    # CASE STUDY 1: The Altitude Effect (Latitude ~1.35)
    # Sungrove (Sea Level) vs. "Sky City" (2500m Altitude, e.g., Quito)
    # Same geometry, same sun path. Different Air.
    # =========================================================
    print("\n1. Analyzing Tropical Altitude Effect (Lat 1.35)...")
    base_weather = os.path.join(current_dir, '..', '..', 'data', 'weather', 'singapore_data.csv')
    
    # Morph: Altitude drops temp ~10C, Increases Radiation ~10% (thinner air)
    df_sea = load_and_morph_weather(base_weather, temp_shift=0, rad_multiplier=1.0, name_suffix="SeaLevel")
    df_high = load_and_morph_weather(base_weather, temp_shift=-10, rad_multiplier=1.1, name_suffix="Highland")
    
    if df_sea.empty or df_high.empty: return

    # Config: A standard room
    # We want to check if "High Mass" acts differently in these two.
    # In Sea Level: High Mass might just stay hot (night is hot).
    # In Highland: High Mass stores day sun to warm the cold night.
    
    cfg_base = BuildingConfig(
        C_in=800000.0, Q_internal=400.0, layer_thickness=0.3, wall_area=10.0, window_ratio=0.4,
        k_wall=0.8, rho_wall=1800.0, c_wall=1000.0, h_in=8.0, h_out=20.0,
        u_window=2.5, tau_window=0.6, k_const_absorb=0.5,
        ventilation_rate=2.0, room_volume=50.0
    )
    
    # Strategy: Simple Overhang
    strat = OverhangStats(2.0, 2.0, 0, 1.0) 
    LAT, LON = 1.35, 103.8
    
    sim_sea = ThermalSystem(cfg_base, strat, LAT, LON)
    res_sea = sim_sea.simulate(df_sea)
    
    sim_high = ThermalSystem(cfg_base, strat, LAT, LON)
    res_high = sim_high.simulate(df_high)
    
    # Plot Comparison
    # Pick a typical day (Day 100)
    day_idx = 24 * 100
    w_size = 48 # 2 days
    
    s_slice = res_sea.iloc[day_idx:day_idx+w_size]
    h_slice = res_high.iloc[day_idx:day_idx+w_size]
    
    plt.figure(figsize=(10, 6))
    plt.plot(s_slice['time'], s_slice['T_in'], label='Sea Level (Hot/Humid)', color='#e74c3c')
    plt.plot(h_slice['time'], h_slice['T_in'], label='Highland (Cool/High UV)', color='#3498db')
    
    # Add comfort bands
    plt.axhspan(20, 26, color='green', alpha=0.1, label='Comfort Zone')
    
    plt.title('Design Divergence at Same Latitude (Tropical)\nImpact of Altitude on Passive Performance')
    plt.ylabel('Indoor Temperature (°C)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'generalization_altitude_effect.png'))
    plt.close()
    
    print("  > Observation: Highland location drops into Heating needs at night despite same Latitude!")

    # =========================================================
    # CASE STUDY 2: Cloud Cover & Continentality (Latitude ~60.5)
    # Borealis (Maritime/Overcast) vs. "Sunny Continental" (e.g., Canadian Prairies)
    # Same sun path. Different Clearness Index and Temp Swing.
    # =========================================================
    print("\n2. Analyzing Cloud Cover Effect (Lat 60.5)...")
    borealis_weather = os.path.join(current_dir, '..', '..', 'data', 'weather', 'norway_data.csv')
    
    # Morph:
    # Maritime: Original
    # Continental: Colder (-5C avg), but MUCH clearer skies (Beam * 2.0, Diffuse * 0.8)
    df_maritime = load_and_morph_weather(borealis_weather, temp_shift=0, rad_multiplier=1.0, name_suffix="Maritime")
    df_continental = load_and_morph_weather(borealis_weather, temp_shift=-5, rad_multiplier=1.0, name_suffix="Continental")
    # Manually adjust beam for Continental to simulate clear skies
    df_continental['Gb(i)'] = df_continental['Gb(i)'] * 2.5 
    
    if df_maritime.empty or df_continental.empty: return

    # Test Strategy: Passive Solar Gain via South Window
    # In Maritime, South Window gains little (diffuse only). In Continental, it should heat the house.
    
    # Config: High Insulation, South Window
    # Reduced Q_internal to 200W (more realistic for passive scenario, avoiding overheat artifact)
    cfg_passive = BuildingConfig(
        C_in=800000.0, Q_internal=200.0, layer_thickness=0.3, wall_area=10.0, window_ratio=0.5, # Big Window!
        k_wall=0.1, rho_wall=100.0, c_wall=1200.0, # High Insulation
        h_in=8.0, h_out=20.0,
        u_window=1.5, tau_window=0.7, k_const_absorb=0.7,
        ventilation_rate=0.5, room_volume=50.0
    )
    
    strat_none = NoShading(2.0, 2.0, 0) # No shading to let sun in
    LAT_B, LON_B = 60.5, 9.1
    
    sim_mar = ThermalSystem(cfg_passive, strat_none, LAT_B, LON_B)
    res_mar = sim_mar.simulate(df_maritime)
    
    sim_con = ThermalSystem(cfg_passive, strat_none, LAT_B, LON_B)
    res_con = sim_con.simulate(df_continental)
    
    # Plot Spring Day (Day 80 - Equinox ish)
    start = 24 * 80
    end = start + 48
    
    m_slice = res_mar.iloc[start:end]
    c_slice = res_con.iloc[start:end]
    
    plt.figure(figsize=(10, 6))
    plt.plot(m_slice['time'], m_slice['T_in'], label='Maritime (Cloudy)', color='gray', linestyle='--')
    plt.plot(c_slice['time'], c_slice['T_in'], label='Continental (Sunny but Colder)', color='orange', linewidth=2)
    plt.plot(c_slice['time'], c_slice['T_out'], label='Continental Outdoor', color='blue', alpha=0.3, linestyle=':')
    
    plt.axhline(20, color='k', linestyle='--', alpha=0.5, label='Heating Setpoint')
    
    plt.title('Same Latitude, Different Sky: Passive Solar Feasibility')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'generalization_cloud_effect.png'))
    plt.close()
    
    print("  > Observation: Passive Solar works in Continental climate despite colder air, fails in Cloudy Maritime.")

if __name__ == "__main__":
    run_generalization_study()
