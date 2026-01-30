import csv
import os
import solar_geometry

# ================= 路径设置 =================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "incidence_angle")
CSV_DIR  = os.path.join(DATA_DIR, "csv")

SECONDS_PER_DAY = 86400

# ================= 工具函数 =================

def write_csv(rel_path, header, rows):
    """
    将数据写入 data/incidence_angle/csv/ 下的相对路径
    """
    filename = os.path.join(CSV_DIR, rel_path)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

# ================= 情况 1：固定地点，连续几天 =================

def generate_daily_variation(lat, lon, start_day, days, time_step):
    rows = []

    start_t = start_day * SECONDS_PER_DAY
    end_t   = start_t + days * SECONDS_PER_DAY

    for t in range(start_t, end_t, time_step):
        beta = solar_geometry.solar_incidence_angle(lat, lon, t)

        rows.append([
            t,
            (t % SECONDS_PER_DAY) / 3600,   # hour of day
            t // SECONDS_PER_DAY,           # day of year
            beta
        ])

    write_csv(
        rel_path=f"daily/daily_lat{lat}_day{start_day}.csv",
        header=["time_sec", "hour", "day_of_year", "incidence_angle"],
        rows=rows
    )

# ================= 情况 2：固定时间，一年内变化 =================

def generate_yearly_at_fixed_time(lat, lon, hour):
    rows = []

    for day in range(365):
        t = day * SECONDS_PER_DAY + hour * 3600
        beta = solar_geometry.solar_incidence_angle(lat, lon, t)
        rows.append([day, beta])

    write_csv(
        rel_path=f"yearly/yearly_lat{lat}_hour{hour}.csv",
        header=["day_of_year", "incidence_angle"],
        rows=rows
    )

# ================= 情况 3：固定时间，不同纬度 =================

def generate_latitude_comparison(day, hour):
    rows = []
    t = day * SECONDS_PER_DAY + hour * 3600

    for lat in range(0, 91, 5):
        beta = solar_geometry.solar_incidence_angle(lat, 0, t)
        rows.append([lat, beta])

    write_csv(
        rel_path=f"latitude_compare/lat_compare_day{day}_hour{hour}.csv",
        header=["latitude", "incidence_angle"],
        rows=rows
    )


    # 连续两天：赤道 & 高纬
generate_daily_variation(lat=0,  lon=0, start_day=180, days=2, time_step=300)
generate_daily_variation(lat=80, lon=0, start_day=180, days=2, time_step=300)

    # 一年变化：正午
generate_yearly_at_fixed_time(lat=0,  lon=0, hour=12)
generate_yearly_at_fixed_time(lat=80, lon=0, hour=12)

    # 纬度比较：夏至正午
generate_latitude_comparison(day=180, hour=12)

print("Solar incidence angle CSV files generated.")
