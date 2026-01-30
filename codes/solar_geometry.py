import numpy as np

# ===== 常量 =====

R_EARTH = 6371000
TILT = np.radians(23.26)

Trot = 24*3600 - 56
Torb = 365*24*3600

a = 1.496e11
e = 0.0167


# ===== 开普勒 =====

def solve_kepler(M, e):
    E = M
    for _ in range(50):
        E = E - (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
    return E


# ===== 主函数：返回入射角 =====

def solar_incidence_angle(
    latitude_deg,
    longitude_deg,
    time_seconds
):

    φ = np.radians(latitude_deg)

    # ---------- 自转 ----------
    λ = np.radians(longitude_deg) + 2*np.pi*time_seconds/Trot

    # ---------- 公转 ----------
    M = 2*np.pi*time_seconds/Torb
    k = solve_kepler(M, e)

    R_dist = a*(1 - e*np.cos(k))

    theta = 2*np.arctan2(
        np.sqrt(1+e)*np.sin(k/2),
        np.sqrt(1-e)*np.cos(k/2)
    )

    # ---------- r 向量 ----------
    r = np.array([
        R_EARTH*(np.cos(λ)*np.cos(φ)*np.cos(TILT)
                 + np.sin(φ)*np.sin(TILT)),

        R_EARTH*(np.sin(λ)*np.cos(φ)),

        R_EARTH*(-np.cos(λ)*np.cos(φ)*np.sin(TILT)
                 + np.sin(φ)*np.cos(TILT))
    ])

    # ---------- R 向量 ----------
    R_vec = np.array([
        R_dist*np.cos(theta),
        R_dist*np.sin(theta),
        0
    ])

    # ---------- 入射角 ----------
    numerator = -np.dot(r, r + R_vec)
    denominator = np.linalg.norm(r) * np.linalg.norm(r + R_vec)

    cos_beta = numerator / denominator

    # 数值安全
    cos_beta = np.clip(cos_beta, -1, 1)

    beta = np.degrees(np.arccos(cos_beta))

    return beta

hours=[0,6,12,18,24,30,36,42,48]
seconds=[1800,1860,1920,1980]
days=[10,30,60,90,180,270]

for s in seconds:
    angle = solar_incidence_angle(0, 0, s)
    print(f"{s}seconds → 入射角 = {angle:.2f}°")
    

for h in hours:
    angle = solar_incidence_angle(0, 0, h*3600)
    print(f"{h}hours → 入射角 = {angle:.2f}°")
    
    
for d in days:
    angle = solar_incidence_angle(0, 0, d*24*3600)
    print(f"{d}days → 入射角 = {angle:.2f}°")
    
    