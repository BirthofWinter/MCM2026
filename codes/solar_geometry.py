import numpy as np


np.set_printoptions(suppress=True, precision=4)
# ===== 常量 =====

R_EARTH = 6371000        # m
D_SUN = 1.496e11         # m
TILT = np.radians(23.44) # 黄赤交角

OMEGA = 2*np.pi / (24*3600)  # 自转角速度


# ===== 模型函数 =====

def reference_point_position(
    latitude_deg,
    time_seconds,
    earth_radius=R_EARTH,
    sun_distance=D_SUN,
    tilt=TILT
):
    """
    返回参考点在日心坐标系中的坐标
    """

    φ = np.radians(latitude_deg)
    θ = OMEGA * time_seconds

    # 地心坐标
    x_p = earth_radius * np.cos(φ) * np.cos(θ)
    y_p = earth_radius * np.cos(φ) * np.sin(θ)
    z_p = earth_radius * np.sin(φ)

    # 倾角旋转
    x = x_p*np.cos(tilt) + z_p*np.sin(tilt)
    y = y_p
    z = -x_p*np.sin(tilt) + z_p*np.cos(tilt)

    # 加上地心位置
    X = sun_distance + x
    Y = y
    Z = z

    return np.array([X, Y, Z])


for h in [0,6,12,18]:
    pos = reference_point_position(40, h*3600)
    print(
        f"{h:2d}h  "
        f"X={pos[0]:.0f}  "
        f"Y={pos[1]:.0f}  "
        f"Z={pos[2]:.0f}"
    )
