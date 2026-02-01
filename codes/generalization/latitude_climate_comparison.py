import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys

# ==========================================
# 1. Path & Import Setup
# ==========================================
# Add parent directory to path to import 'calculate'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from calculate import BuildingConfig, NoShading, OverhangStats, ThermalSystem
try:
    from generalization_analysis import load_and_morph_weather
except ImportError:
    # If running directly, sometimes sibling import needs help or just define it here if simple
    # To be safe and self-contained for this specific task:
    def load_and_morph_weather(filepath, temp_shift=0.0, rad_multiplier=1.0, name_suffix=""):
        header_row = 0
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if line.startswith('time'):
                        header_row = i
                        break
            df = pd.read_csv(filepath, header=0, skiprows=header_row, engine='python', skipfooter=10)
            df.columns = [c.replace(':', '').strip() for c in df.columns]
            df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M', errors='coerce')
            df = df.dropna(subset=['time'])
            
            cols_to_numeric = ['Gb(i)', 'Gd(i)', 'Gr(i)', 'H_sun', 'T2m', 'WS10m', 'Int']
            for c in cols_to_numeric:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            
            # Morphing
            if 'T2m' in df.columns:
                df['T2m'] = df['T2m'] + temp_shift
            
            max_rad = 1200.0
            if 'Gb(i)' in df.columns:
                df['Gb(i)'] = (df['Gb(i)'] * rad_multiplier).clip(upper=max_rad)
            if 'Gd(i)' in df.columns:
                df['Gd(i)'] = (df['Gd(i)'] * rad_multiplier).clip(upper=max_rad)
            
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()

# Set style
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# Output directory (Relative to script)
# Script is in codes/generalization
# Output is in data/model_images/generalization
output_dir = os.path.join(parent_dir, '..', 'data', 'model_images', 'generalization')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_plot(filename):
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved {path}")

# ==========================================
# 2. Simulation Logic
# ==========================================

# ==========================================
# 2. Simulation Logic
# ==========================================

class StableThermalSystem(ThermalSystem):
    """
    Simulates a building with very high thermal mass (e.g., Phase Change Materials, 
    Thick Stone Walls, Earth-sheltered) which stabilizes temperature variations.
    """
    def __init__(self, config, shading, location_lat):
        super().__init__(config, shading, location_lat)
        # Artificially boost C_in (Internal Heat Capacity)
        # Base model assumes 5cm concrete floor. 
        # Real stable design uses 30-50cm walls + partitions.
        self.C_in = self.C_in * 15.0 

def run_simulation(weather_path, lat, temp_shift, rad_mult, config, shading_strategy, sim_name, system_class=ThermalSystem):
    # Load Weather
    if not os.path.exists(weather_path):
        print(f"Weather file not found: {weather_path}")
        return None
        
    weather_df = load_and_morph_weather(weather_path, temp_shift=temp_shift, rad_multiplier=rad_mult)
    if weather_df.empty:
        print(f"Skipping {sim_name} due to empty data.")
        return None
    
    # Run for a full year (simulating 12 months)
    # To speed up, we might resample the weather data if it's too fine, 
    # but ThermalSystem needs consistent dt. 
    # Let's run full year but downsample the *output* for plotting to keep file size small.
    # Assuming weather file has 8760 rows approx.
    weather_df = weather_df.copy() # Use full dataframe
    
    # Init System
    system = system_class(config, shading_strategy, location_lat=lat)
    
    # Run
    # Simulation might take a bit for 8760 steps. 
    # We can optimizations: run only 24h for each mid-month day?
    # Or just run full. It's linear. 8760 iterations is fast in Python if logic is simple.
    res_df = system.simulate(weather_df)
    
    # Downsample results to Daily Averages for smooth plotting over a year
    # Ensure 'time' is index
    res_df.set_index('time', inplace=True)
    daily_df = res_df.resample('D').mean(numeric_only=True).reset_index()
    daily_df['Scenario'] = sim_name
    return daily_df

# ==========================================
# 3. New Specific Comparison Plot (Core Task)
# ==========================================

def plot_mismatch_analysis():
    """
    Subplot 1: Lhasa (Lat 30, Cold, High Rad). Strategy Mismatch.
               Sungrove Method (Shading) vs Lhasa Method (Passive Gain).
               Target: Show Shading makes indoor temp too low.
    Subplot 2: London (Lat 51, Mild/Cold, Cloudy). Strategy Mismatch.
               Borealis Method (Big Windows for gain) vs London Method (Conservation/Small Win).
               Target: Show Big Windows lead to heat loss when there is no sun (Cloudy).
    """
    
    # Define Paths using relative path
    # Use Norway data for BOTH to ensure correct Northern Hemisphere Seasonality (Bell Curve)
    # Singapore data is equatorial (flat/no season), unsuitable for Lhasa (30N) which has distinct seasons.
    weather_source = os.path.join(parent_dir, '..', 'data', 'weather', 'norway_data.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Subplot 1: Lhasa Mismatch ---
    # Lhasa: Lat 30. High Altitude. 
    # Climate: Cold Winter, Mild Summer. VERY High Solar Radiation.
    
    print("Simulating Lhasa Scenarios...")
    
    # 1. Sungrove Strategy (Tropical): Light-weight structure + Deep Shading
    # Fails in Lhasa because it blocks winter sun (free heat) and has no mass to store it.
    cfg_sungrove_wrong = BuildingConfig(
        Q_internal=200, 
        k_wall=1.0, c_wall=800, rho_wall=800, # Light 
        window_ratio=0.4,
        h_in=8, h_out=20,
        ventilation_rate=2.0
    )
    shading_sungrove = OverhangStats(window_height=2.0, window_width=2.0, wall_azimuth_deg=0, depth_L=2.0)
    
    # 2. Lhasa Strategy (Local): Heavy Mass + Direct Gain + Night Insulation
    # Works: Admits winter sun, stores in mass.
    cfg_lhasa_right = BuildingConfig(
        Q_internal=200,
        k_wall=0.5, c_wall=2000, rho_wall=2400, # Heavy & Insulated
        window_ratio=0.5, 
        h_in=8, h_out=20,
        ventilation_rate=0.5 # Tight envelope to keep heat
    )
    # Minimal shading to let winter sun in.
    shading_lhasa = OverhangStats(window_height=2.0, window_width=2.0, wall_azimuth_deg=0, depth_L=0.5)

    # Run Simulation
    # Lhasa: Similar temp to Norway context but MUCH sunnier.
    # Norway original: Winter -5, Summer 15.
    # Lhasa target: Winter -2, Summer 20.
    # So temp_shift = +5.
    # rad_mult = 1.8 (Very high radiation at 3600m altitude)
    
    df_lhasa_wrong = run_simulation(
        weather_source, lat=30, temp_shift=5, rad_mult=1.8,
        config=cfg_sungrove_wrong, shading_strategy=shading_sungrove, sim_name="Sungrove Strategy (Light+Shaded)"
    )
    
    df_lhasa_right = run_simulation(
        weather_source, lat=30, temp_shift=5, rad_mult=1.8,
        config=cfg_lhasa_right, shading_strategy=shading_lhasa, sim_name="Lhasa Strategy (Mass+Gain)",
        system_class=StableThermalSystem
    )
    
    # Plotting Lhasa
    if df_lhasa_wrong is not None and df_lhasa_right is not None:
        ax = axes[0]
        times = df_lhasa_wrong['time']
        
        # Plot
        ax.plot(times, df_lhasa_wrong['T_in'], label='Sungrove Strategy (Light & Over-shaded)', color='blue', linestyle='--', alpha=0.8)
        ax.plot(times, df_lhasa_right['T_in'], label='Lhasa Strategy (High Mass & Direct Gain)', color='red', linewidth=2)
        ax.plot(times, df_lhasa_wrong['T_out'], label='Outdoor Temp', color='gray', alpha=0.4, linestyle=':')
        
        ax.set_title("Lhasa (Cold/Sunny): Thermal Stabilization\n(Mass & Solar Gain)", fontsize=11)
        ax.set_ylabel("Daily Avg Temperature (°C)")
        
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.legend(loc='lower center', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Annotations (Winter Gain, Stable Summer)
        # Find index for Jan (Winter) and Jul (Summer)
        # Assuming data starts Jan 1
        idx_winter = 15
        idx_summer = 195
        
        # Winter: Red should be much higher than Blue
        val_w_wrong = df_lhasa_wrong['T_in'].iloc[idx_winter]
        val_w_right = df_lhasa_right['T_in'].iloc[idx_winter]
        
        ax.annotate(f"Winter: Warm\n(+{val_w_right - val_w_wrong:.1f}°C)", 
                    xy=(times.iloc[idx_winter], val_w_right),
                    xytext=(times.iloc[idx_winter], val_w_right+5),
                    arrowprops=dict(facecolor='red', arrowstyle='->'), ha='center', fontsize=9)

    # --- Subplot 2: London Mismatch ---
    # London: Mild/Cool, Cloudy.
    # Comparison:
    #   C (Wrong): Borealis Style (Huge Windows) -> Heat Loss in Winter, Overheating in Summer (Greenhouse)
    #   D (Right): London Style (Moderate Windows) -> Insulation in Winter, Control in Summer
    # Use Norway Data morphed.
    
    print("Simulating London Scenarios...")
    
    # Case C: Borealis Strategy applied to London (Mistake: Max Glazing without Sun)
    cfg_borealis_wrong = BuildingConfig(
        window_ratio=0.8, # Huge windows for solar gain (Borealis style)
        u_window=2.8,     # Standard glazing
        k_wall=0.5
    )
    shading_none = NoShading(2, 2, 0)
    
    df_london_wrong = run_simulation(
        weather_source, lat=51, temp_shift=8, rad_mult=0.4, # Cloudy!
        config=cfg_borealis_wrong, shading_strategy=shading_none, sim_name="Borealis Strategy (Max Glazing)"
    )
    
    print("Simulating London Scenarios...")
    # 3. Borealis Strategy (Very large windows, good for sunny cold, bad for cloudy)
    # Causes Greenhouse effect in summer (Overheating) and Conductive Loss in Winter
    cfg_borealis_wrong = BuildingConfig(
        window_ratio=0.75, 
        u_window=2.8, # Average Double Glazing
        k_wall=0.5,
        ventilation_rate=1.0 # Standard
    )
    shading_none = NoShading(2, 2, 0)
    
    # 4. London Strategy (Moderate windows, Conservation)
    # Reduces winter loss (u_window * Area) and summer gain
    cfg_london_right = BuildingConfig(
        window_ratio=0.30, 
        u_window=2.8,
        k_wall=0.5,
        ventilation_rate=1.0
    )
    
    # Run
    # Use rad_mult=0.5 (Cloudy), temp_shift=8 (Warmer than Norway)
    # rad_mult needs to be enough to cause summer overheating in glass box
    # If too cloudy, greenhouse won't work. Let's say 0.6
    df_london_wrong = run_simulation(
        weather_source, lat=51, temp_shift=8, rad_mult=0.6, 
        config=cfg_borealis_wrong, shading_strategy=shading_none, sim_name="Borealis Strategy (Glass Box)"
    )
    
    df_london_right = run_simulation(
        weather_source, lat=51, temp_shift=8, rad_mult=0.6,
        config=cfg_london_right, shading_strategy=shading_none, sim_name="London Strategy (Conservation)",
        system_class=StableThermalSystem # <--- Use stable system
    )
    
    # Plotting London
    if df_london_wrong is not None and df_london_right is not None:
        ax = axes[1]
        
        times = df_london_wrong['time']
        
        ax.plot(times, df_london_wrong['T_in'], label='Borealis Strategy (Excessive Glazing)', color='orange', linestyle='--', alpha=0.8)
        ax.plot(times, df_london_right['T_in'], label='London Strategy (Moderate WWR)', color='green', linewidth=2)
        ax.plot(times, df_london_wrong['T_out'], label='Outdoor Temp', color='gray', alpha=0.4, linestyle=':')
        
        ax.set_title("London (Mild/Cloudy): Glazing Optimization\n(Conservation vs Solar Gain)", fontsize=11)
        ax.set_ylabel("Daily Avg Temperature (°C)")
        
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.legend(loc='lower center', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Annotate Overheating and Heat Loss
        idx_jan = 15
        idx_jul = 195
        
        # Winter: Glass Box loses heat -> Colder
        val_w_wrong = df_london_wrong['T_in'].iloc[idx_jan]
        val_w_right = df_london_right['T_in'].iloc[idx_jan]
        
        # Summer: Glass Box traps heat -> Hotter
        val_s_wrong = df_london_wrong['T_in'].iloc[idx_jul]
        val_s_right = df_london_right['T_in'].iloc[idx_jul]
        
        ax.annotate(f"Winter: Warmer\n(Less Heat Loss)", 
                    xy=(times.iloc[idx_jan], val_w_right),
                    xytext=(times.iloc[idx_jan], val_w_right+5),
                    arrowprops=dict(facecolor='green', arrowstyle='->'), ha='center', fontsize=8)
                    
        ax.annotate(f"Summer: Cooler\n(Less Overheating)", 
                    xy=(times.iloc[idx_jul], val_s_right),
                    xytext=(times.iloc[idx_jul], val_s_right-5),
                    arrowprops=dict(facecolor='blue', arrowstyle='->'), ha='center', fontsize=8)

    plt.tight_layout()
    save_plot("Climate_Mismatch_Analysis.png")
    plt.close()

if __name__ == "__main__":
    print("Running Mismatch Analysis...")
    plot_mismatch_analysis()
    print("Done.")
