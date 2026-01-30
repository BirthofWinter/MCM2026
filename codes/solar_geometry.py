import numpy as np


np.set_printoptions(suppress=True, precision=4)
# ===== 常量 =====

R_EARTH = 6371000        # m 地球半径
D_SUN = 1.496e11         # m 地日距离
TILT = np.radians(23.26) # 黄赤交角
YITA = np.radians(13.26) # 冬至远日夹角
Trot = 24*3600 # 地球自转周期
Torb = 365*24*3600 # 地球公转周期

OMEGA = 2*np.pi / (Trot)  # 自转角速度


# ===== 模型函数 =====

def reference_point_position(
    latitude_deg,   # 纬度
    longitude_deg,  # 经度
    time_seconds,   # 经过时间
    earth_radius=R_EARTH,
    sun_distance=D_SUN,
    tilt=TILT,
    yita=YITA,
    Trot=Trot,
    Torb=Torb,
):
    """
    返回参考点在日心坐标系中的坐标
    """

    r_x=earth_radius * (np.cos(longitude_deg) * np.cos(latitude_deg) * np.cos(tilt)+ np.sin(latitude_deg)*np.sin(tilt))
    r_y=earth_radius * np.sin(longitude_deg) * np.sin(latitude_deg)
    r_z=earth_radius * (- np.cos(longitude_deg)*np.cos(latitude_deg)*np.sin(tilt)+np.sin(latitude_deg)*np.cos(tilt))

    R_x=sun_distance*np.cos(theta)
    R_y=sun_distance*np.sin(theta)
    R_z=0
    
    return np.array([X, Y, Z])


for h in [0,6,12,18]:
    pos = reference_point_position(40, 60 ,h*3600)
    print(
        f"{h:2d}h  "
        f"X={pos[0]:.0f}  "
        f"Y={pos[1]:.0f}  "
        f"Z={pos[2]:.0f}"
    )
