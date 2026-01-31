import plot_data as pld
import generate_data as gd

gd.generate_daily_variation(lat=5, lon=0, start_day=30, days=30, time_step=1800)
pld.plot_daily_variation(lat=5, start_day=30)
pld.plot_daily_histogram(lat=5, start_day=30)