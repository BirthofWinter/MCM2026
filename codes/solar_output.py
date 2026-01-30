import csv
import solar_geometry


latitude = 1.3
longitude = 0

time_step = 300          # 每5分钟采样
total_days = 2

total_time = total_days * 24 * 3600

# ===== 写CSV =====

with open("solar_angles_singapore_2days.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "time_seconds",
        "time_hours",
        "day",
        "incidence_angle_deg",
        "is_daylight"
    ])

    for t in range(0, total_time, time_step):

        beta = solar_geometry.solar_incidence_angle(latitude, longitude, t)

        is_day = beta < 90

        writer.writerow([
            f"{t:.0f}",
            f"{t/3600:.2f}",
            f"{t/86400:.0f}",
            f"{beta:.2f}",
            int(is_day)
        ])

print("CSV 导出完成")