import os
import pandas as pd
import matplotlib.pyplot as plt

# ================= 路径设置 =================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "incidence_angle")
CSV_DIR  = os.path.join(DATA_DIR, "csv")
FIG_DIR  = os.path.join(DATA_DIR, "figs")

# ================= 情况 1：连续两天变化 =================

def plot_daily_variation(lat, start_day):
    csv_path = os.path.join(
        CSV_DIR, f"daily/daily_lat{lat}_day{start_day}.csv"
    )
    df = pd.read_csv(csv_path)

    # 将时间展开为连续小时
    time_hours = df["hour"] + 24 * (df["day_of_year"] - start_day)

    plt.figure(figsize=(8, 4))
    plt.plot(time_hours, df["incidence_angle"])
    plt.xlabel("Time (hours)")
    plt.ylabel("Incidence Angle (deg)")
    plt.title(f"Solar Incidence Angle (Lat = {lat}°)")
    plt.grid()

    out_dir = os.path.join(FIG_DIR, "daily")
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(
        os.path.join(out_dir, f"daily_lat{lat}_day{start_day}.png"),
        dpi=300
    )
    plt.close()

# ================= 情况 2：一年变化 =================

def plot_yearly_variation(lat, hour):
    csv_path = os.path.join(
        CSV_DIR, f"yearly/yearly_lat{lat}_hour{hour}.csv"
    )
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 4))
    plt.plot(df["day_of_year"], df["incidence_angle"])
    plt.xlabel("Day of Year")
    plt.ylabel("Incidence Angle (deg)")
    plt.title(f"Yearly Variation at {lat}° Latitude (Hour = {hour})")
    plt.grid()

    out_dir = os.path.join(FIG_DIR, "yearly")
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(
        os.path.join(out_dir, f"yearly_lat{lat}_hour{hour}.png"),
        dpi=300
    )
    plt.close()

# ================= 情况 3：纬度比较 =================

def plot_latitude_comparison(day, hour):
    csv_path = os.path.join(
        CSV_DIR, f"latitude_compare/lat_compare_day{day}_hour{hour}.csv"
    )
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 4))
    plt.plot(df["latitude"], df["incidence_angle"], marker="o")
    plt.xlabel("Latitude (deg)")
    plt.ylabel("Incidence Angle (deg)")
    plt.title(f"Latitude Dependence (Day {day}, Hour {hour})")
    plt.grid()

    out_dir = os.path.join(FIG_DIR, "latitude_compare")
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(
        os.path.join(out_dir, f"lat_compare_day{day}_hour{hour}.png"),
        dpi=300
    )
    plt.close()



plot_daily_variation(lat=0,  start_day=180)
plot_daily_variation(lat=80, start_day=180)

plot_yearly_variation(lat=0,  hour=12)
plot_yearly_variation(lat=80, hour=12)

plot_latitude_comparison(day=180, hour=12)

print("Figures generated.")
