import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def plot_daily_histogram(lat, start_day):
    csv_path = os.path.join(
        CSV_DIR, f"daily/daily_lat{lat}_day{start_day}.csv"
    )
    df = pd.read_csv(csv_path)

    data = df[df["incidence_angle"] < 90]["incidence_angle"]

    plt.figure(figsize=(6, 4))

    counts, bins, patches = plt.hist(
        data,
        bins=30,
        range=(0, 90),
        edgecolor="black",
        color="lightgray"
    )

    # ===== Step 1: 找主峰 bin =====
    max_count = counts.max()
    threshold = 0.9 * max_count

    peak_indices = [
        i for i, c in enumerate(counts) if c >= threshold
    ]

    # ===== Step 2: 连续 bin 分组 =====
    peak_groups = []
    current_group = [peak_indices[0]]

    for idx in peak_indices[1:]:
        if idx == current_group[-1] + 1:
            current_group.append(idx)
        else:
            peak_groups.append(current_group)
            current_group = [idx]

    peak_groups.append(current_group)

    # ===== Step 3: 对每个峰单独处理 =====
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]  # 红、橙、绿
    representative_angles = []

    for k, group in enumerate(peak_groups):
        centers = []
        for i in group:
            patches[i].set_facecolor(colors[k % len(colors)])
            centers.append(0.5 * (bins[i] + bins[i + 1]))

        rep_angle = np.median(centers)
        representative_angles.append(rep_angle)

        plt.axvline(
            rep_angle,
            linestyle="--",
            color=colors[k % len(colors)],
            linewidth=1.2,
            label=f"Peak {k+1}: {rep_angle:.1f}°"
        )

    # ===== 坐标轴与样式 =====
    plt.xlim(0, 90)
    plt.xticks(range(0, 91, 10))
    plt.xlabel("Incidence Angle (deg)")
    plt.ylabel("Frequency")
    plt.title(f"Incidence Angle Distribution (Lat={lat}°, 2 Days)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.join(FIG_DIR, "daily_hist")
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(
        os.path.join(out_dir, f"daily_hist_lat{lat}_day{start_day}.png"),
        dpi=300
    )
    plt.close()

    return representative_angles



    
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


def plot_yearly_histogram(lat, hour):
    csv_path = os.path.join(
        CSV_DIR, f"yearly/yearly_lat{lat}_hour{hour}.csv"
    )
    df = pd.read_csv(csv_path)

    data = df[df["incidence_angle"] < 90]["incidence_angle"]

    plt.figure(figsize=(6, 4))

    counts, bins, patches = plt.hist(
        data,
        bins=30,
        range=(0, 90),
        edgecolor="black",
        color="lightgray"
    )

    # ===== Step 1: 找主峰 bin =====
    max_count = counts.max()
    threshold = 0.9 * max_count

    peak_indices = [
        i for i, c in enumerate(counts) if c >= threshold
    ]

    # 防止极端情况（理论上不该发生，但稳妥）
    if not peak_indices:
        plt.close()
        return []

    # ===== Step 2: 连续 bin 分组 =====
    peak_groups = []
    current_group = [peak_indices[0]]

    ANGLE_GAP_THRESHOLD = 4.0  # degrees

    for idx in peak_indices[1:]:
        prev_idx = current_group[-1]

        prev_center = 0.5 * (bins[prev_idx] + bins[prev_idx + 1])
        curr_center = 0.5 * (bins[idx] + bins[idx + 1])

        if abs(curr_center - prev_center) <= ANGLE_GAP_THRESHOLD:
            current_group.append(idx)
        else:
            peak_groups.append(current_group)
            current_group = [idx]

    peak_groups.append(current_group)

    # ===== Step 3: 对每个峰单独处理 =====
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]  # 红、橙、绿
    representative_angles = []

    for k, group in enumerate(peak_groups):
        centers = []
        for i in group:
            patches[i].set_facecolor(colors[k % len(colors)])
            centers.append(0.5 * (bins[i] + bins[i + 1]))

        rep_angle = np.median(centers)
        representative_angles.append(rep_angle)

        plt.axvline(
            rep_angle,
            linestyle="--",
            color=colors[k % len(colors)],
            linewidth=1.2,
            label=f"Peak {k+1}: {rep_angle:.1f}°"
        )

    # ===== 坐标轴与样式 =====
    plt.xlim(0, 90)
    plt.xticks(range(0, 91, 10))
    plt.xlabel("Incidence Angle (deg)")
    plt.ylabel("Frequency")
    plt.title(
        f"Yearly Incidence Angle Distribution\n"
        f"(Lat={lat}°, Hour={hour})"
    )
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.join(FIG_DIR, "yearly_hist")
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(
        os.path.join(out_dir, f"yearly_hist_lat{lat}_hour{hour}.png"),
        dpi=300
    )
    plt.close()

    return representative_angles




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
    
def plot_latitude_interval_statistics(day, hour):
    csv_path = os.path.join(
        CSV_DIR, f"latitude_compare/lat_compare_day{day}_hour{hour}.csv"
    )
    df = pd.read_csv(csv_path)

    bins = [0, 30, 60, 90]
    labels = ["0–30°", "30–60°", "60–90°"]

    df = df[df["incidence_angle"] < 90]
    df["angle_bin"] = pd.cut(
        df["incidence_angle"],
        bins=bins,
        labels=labels
    )

    counts = df["angle_bin"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", edgecolor="black")
    plt.xlabel("Incidence Angle Interval")
    plt.ylabel("Count")
    plt.title(f"Angle Interval Statistics (Day {day}, Hour {hour})")
    plt.grid(axis="y", alpha=0.3)

    out_dir = os.path.join(FIG_DIR, "latitude_compare_hist")
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(
        os.path.join(
            out_dir, f"lat_interval_day{day}_hour{hour}.png"
        ),
        dpi=300
    )
    plt.close()

    
    




plot_daily_variation(lat=0,  start_day=180)
plot_daily_variation(lat=80, start_day=180)

plot_daily_variation(lat=5, start_day=0)

plot_daily_histogram(lat=0,  start_day=180)
plot_daily_histogram(lat=80, start_day=180)

plot_daily_histogram(lat=5, start_day=0)


plot_yearly_variation(lat=0,  hour=12)
plot_yearly_variation(lat=0,  hour=8)
plot_yearly_variation(lat=0,  hour=14)
plot_yearly_variation(lat=23,  hour=14)
plot_yearly_variation(lat=0,  hour=24)
plot_yearly_variation(lat=80, hour=12)
plot_yearly_variation(lat=80,  hour=24)

plot_yearly_histogram(lat=0,  hour=12)
plot_yearly_histogram(lat=0,  hour=8)
plot_yearly_histogram(lat=23,  hour=14)
plot_yearly_histogram(lat=23,  hour=0)
plot_yearly_histogram(lat=80, hour=12)
plot_yearly_histogram(lat=0,  hour=24)
plot_yearly_histogram(lat=80, hour=24)
plot_yearly_histogram(lat=0,  hour=14)

plot_latitude_comparison(day=180, hour=12)
plot_latitude_interval_statistics(day=180, hour=12)

print("Figures generated.")
