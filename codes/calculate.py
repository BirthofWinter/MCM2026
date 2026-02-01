import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod

# ==========================================
# 1. 配置与参数类
# ==========================================

@dataclass
class BuildingConfig:
    """建筑物理参数配置"""
    # 几何参数 (用于光照和热容归一化)
    module_width: float = 4.0   # [m] 模拟模块宽度
    room_depth: float = 8.0     # [m] 房间深度
    room_height: float = 3.5    # [m] 房间层高
    
    # 基础热工属性
    Q_internal: float = 500.0   # [W] 内部发热量 (人员+设备)
    window_ratio: float = 0.4   # [-] 窗墙比 (WWR)
    
    # 墙体材料属性
    layer_thickness: float = 0.3 # [m] 墙体厚度
    k_wall: float = 0.8         # [W/mK] 导热系数 (Concrete)
    rho_wall: float = 1800.0    # [kg/m3] 密度
    c_wall: float = 1000.0      # [J/kgK] 比热容
    
    # 对流系数
    h_in: float = 8.0           # [W/m2K] 内表面对流
    h_out: float = 20.0         # [W/m2K] 外表面对流
    
    # 派生属性将基于以上参数计算
    
    # 窗户属性
    u_window: float = 2.8 # [W/(m^2*K)] 窗户传热系数 (U-value)
    tau_window: float = 0.75 # [-] 窗户太阳能总透射率 (SHGC approx, 用于热)
    tau_visible: float = 0.6 # [-] 窗户可见光透射率 (VLT, 用于光)
    
    # 新增：通风属性
    ventilation_rate: float = 1.0 # [ACH] 每小时换气次数 Air Changes per Hour
    
    # 光照计算属性
    c_room_light: float = 0.5 # [-] 室内光反射系数 (Light Coefficient) - 调低一点，考虑深处衰减

    # 高级策略开关
    night_cooling: bool = False # 是否开启夜间通风
    night_vent_rate: float = 5.0 # [ACH] 夜间通风换气次数

    # 辐射吸收系数
    k_const_absorb: float = 0.6 # [-] Give default value to fix valid parameter order
    
    @property
    def wall_area(self):
        """外墙总面积 (模块)"""
        return self.module_width * self.room_height

    @property
    def floor_area(self):
        """地板面积 (模块)"""
        return self.module_width * self.room_depth

    @property
    def room_volume(self):
        """房间体积 (模块)"""
        return self.floor_area * self.room_height
        
    @property
    def window_area(self):
        return self.wall_area * self.window_ratio

    @property
    def solid_wall_area(self):
        return self.wall_area * (1 - self.window_ratio)

# ==========================================
# 2. 太阳模组
# ==========================================

import solar_geometry # 导入新的高精度物理模型

class SolarModel:
    """计算太阳位置 (使用 solar_geometry 中的开普勒轨道模型)"""
    def __init__(self, lat_deg, lon_deg=0):
        self.lat_deg = lat_deg 
        self.lon_deg = lon_deg
        
    def get_position(self, day_of_year: int, hour_of_day_utc: float):
        """
        计算太阳高度角(elevation)和方位角(azimuth)
        return: theta1 (elevation, rad), theta2 (azimuth, rad, South=0)
        """
        # 直接调用 solar_geometry 中的高精度计算函数
        theta1, theta2 = solar_geometry.get_solar_position(
            day_of_year, hour_of_day_utc, self.lat_deg, self.lon_deg
        )
        return theta1, theta2

# ==========================================
# 3. 遮阳策略 (核心需求：不同遮阳类型)
# ==========================================

class ShadingStrategy(ABC):
    def __init__(self, window_height, window_width, wall_azimuth_deg):
        self.H = window_height
        self.W = window_width
        self.gamma = np.radians(wall_azimuth_deg) # 墙面朝向 (0=South)

    @abstractmethod
    def calculate_shade_factor(self, theta1_elev, theta2_azim):
        """返回遮挡比例 F_shade (0.0 - 1.0)"""
        pass

class NoShading(ShadingStrategy):
    def calculate_shade_factor(self, theta1_elev, theta2_azim):
        return 0.0

class OverhangStats(ShadingStrategy):
    """水平遮阳板 (Overhang)"""
    def __init__(self, window_height, window_width, wall_azimuth_deg, depth_L):
        super().__init__(window_height, window_width, wall_azimuth_deg)
        self.L = depth_L
        
    def calculate_shade_factor(self, theta1_elev, theta2_azim):
        if theta1_elev <= 0: return 1.0 # 没太阳即全遮 (或者说无光)
        
        # 相对方位角 beta = |theta2 - gamma|
        rel_azimuth = theta2_azim - self.gamma
        
        # 如果太阳在墙的背面，则完全遮挡
        if np.cos(rel_azimuth) <= 0:
            return 1.0
            
        # 阴影垂直长度
        shadow_h = self.L * np.tan(theta1_elev) / np.cos(rel_azimuth)
        
        f_shade = np.clip(shadow_h / self.H, 0.0, 1.0)
        return f_shade

class VerticalFins(ShadingStrategy):
    """垂直遮阳板 (Vertical Fins)"""
    def __init__(self, window_height, window_width, wall_azimuth_deg, depth_L, count=2):
        super().__init__(window_height, window_width, wall_azimuth_deg)
        self.L = depth_L
        self.count = count # Fin count (e.g., one left, one right, or multiple)
        
    def calculate_shade_factor(self, theta1_elev, theta2_azim):
        if theta1_elev <= 0: return 1.0
        
        # 相对方位角 beta
        rel_azimuth = theta2_azim - self.gamma
        
        # 垂直遮阳取决于方位角差
        # 阴影长度在水平方向上的投影: L * tan(|rel_azimuth|) ? 
        # No, geometry is: L * tan(rel_azimuth) is horizontal displacement on wall?
        # Correct projection: L * tan(|rel_azimuth|) only if sun is 'behind' the fin relative to window normal?
        # Simple model: Fin is perpendicular to wall.
        # Shadow width W_shadow = L * tan(|rel_azimuth|)
        # Shaded area fraction depends on Fin placement. 
        # Assuming fins are closely spaced or just side-fins.
        # Let's assume infinite vertical fins with spacing S?
        # Simplified: W_shadow = L * |tan(rel_azimuth)|
        # F_shade = clip(W_shadow / W_separation, 0, 1) or similar.
        # For a single window with side fins:
        # If sun is from left, left fin casts shadow.
        
        # Let's simply implement Side Fins logic.
        shadow_w = self.L * np.abs(np.tan(rel_azimuth))
        f_shade = np.clip(shadow_w / self.W, 0.0, 1.0)
        
        return f_shade

# ==========================================
# 4. 热模型 (PDE Solver)
# ==========================================

class ThermalSystem:
    def __init__(self, config: BuildingConfig, shading: ShadingStrategy, location_lat, location_lon=0):
        self.cfg = config
        self.shade = shading
        self.solar = SolarModel(location_lat, location_lon)
        
        # 初始地板面积
        if self.cfg.floor_area <= 0:
            self.cfg.floor_area = self.cfg.room_volume / 3.0

        # PDE 网格初始化
        self.N_nodes = 10
        self.dx = self.cfg.layer_thickness / self.N_nodes
        self.alpha = self.cfg.k_wall / (self.cfg.rho_wall * self.cfg.c_wall)
        
        # 状态变量初始化
        self.T_wall = np.ones(self.N_nodes + 1) * 20.0 # 初始墙温
        self.T_in = 20.0 # 初始室温
        
        # 时间步长 (s)
        # Stability criteria for Explicit FDM with Convection Boundary
        # Fo = alpha * dt / dx^2
        # Bi = h * dx / k
        # Stability: Fo * (1 + Bi) <= 0.5
        
        Bi_out = self.cfg.h_out * self.dx / self.cfg.k_wall
        Bi_in = self.cfg.h_in * self.dx / self.cfg.k_wall
        Bi_max = max(Bi_out, Bi_in)
        
        limit_dt_internal = self.dx**2 / (2 * self.alpha)
        limit_dt_boundary = self.dx**2 / (2 * self.alpha * (1 + Bi_max))
        
        limit_dt = min(limit_dt_internal, limit_dt_boundary)
        
        # 限制最大步长为 1800s (30分钟)，加快计算速度
        self.dt = min(1800, int(limit_dt * 0.95)) 
        # 确保至少非零
        self.dt = max(1, self.dt) 
        
        # 自动计算 C_in (Internal Thermal Mass)
        # Air
        c_air = 1.225 * 1005 * self.cfg.room_volume
        # Mass (Floor + Ceiling = 2 * FloorArea)
        # Concrete: 2000 kg/m3 * 880 J/kgK
        # Effective depth 0.05m participating in daily swing
        c_mass = 2 * self.cfg.floor_area * 0.05 * 2000 * 880
        self.C_in = c_air + c_mass

    def step(self, t_current, weather_row):
        """执行一个时间步长的模拟"""
        T_out = weather_row['T2m']
        # 假设 Gb(i) 为水平面直射辐射
        I_horizontal_beam = weather_row.get('Gb(i)', 0.0)
        
        doy = t_current.dayofyear
        # Use UTC hour from timestamp
        hour = t_current.hour + t_current.minute/60.0
        theta1, theta2 = self.solar.get_position(doy, hour)
        
        f_shade = self.shade.calculate_shade_factor(theta1, theta2)
        
        rel_azimuth = theta2 - self.shade.gamma
        cos_rel_az = max(0, np.cos(rel_azimuth))
        
        # 墙面/窗户接收到的有效通量 (W/m2)
        # 根据 formulas.md 近似: I_term = I0 * sin(theta1) * cos(theta2 - gamma)
        # 如果 I_input = I0 * sin(theta1) (水平面), 则 I_term = I_input * cos(theta2 - gamma)
        solar_flux_base = I_horizontal_beam * cos_rel_az 
        
        # 光照度计算 (Illuminance) - 单位 Lux
        # I_vis0 (Lux) approx Solar_Flux (W/m2) * Luminous Efficacy (lm/W)
        # 太阳直射光效约 105 lm/W
        EFFICACY = 105.0 
        I_vis0 = abs(solar_flux_base) * EFFICACY
        
        # Formula: Ein = Ivis0 * (eta * S / Afloor) * tau_vis * (1 - Fshade) * Croom
        # Note: solar_flux_base already includes cos(theta2 - gamma) * sin(theta1) implicitly via I_horiz * cos_rel / sin_elev?
        # Check: I_horizontal = I0 * sin(elev). I_vertical_surface = I0 * cos(inc_angle).
        # solar_flux_base definition above: I_horizontal_beam * cos_rel_az
        # Actually, Calculate logic above is simplified. I_horizontal_beam is usually I_beam_normal * sin(theta1).
        # So I_beam_normal = I_horizontal_beam / sin(theta1).
        # Flux on wall = I_beam_normal * cos(incident_angle).
        # Incident angle approx: cos(inc) = cos(elev)*cos(rel_az).
        # So Flux = (I_horiz / sin_elev) * cos_elev * cos_rel_az = I_horiz * cot_elev * cos_rel_az.
        # But previous logic was: solar_flux_base = I_horizontal_beam * cos_rel_az. This is approximate for low elevation, high error for high elevation.
        # Let's fix flux base calculation first for better physics if possible, assuming I_horizontal_beam is Gb(i).
        
        # Better Flux Approximation:
        # I0 in formulas.md corresponds to DNI (Direct Normal Irradiance)
        # Avoid div by zero for DNI estimate
        sin_e_safe = max(0.05, np.sin(theta1))
        # Direct Normal Irradiance (DNI) ~ I_horizontal / sin(elevation)
        I0 = I_horizontal_beam / sin_e_safe
        
        # Wall Incidence Cosine (Vertical Surface)
        # cos(theta_inc) = cos(theta1) * cos(theta2 - gamma)
        # Matches formula: I0 * cos(theta1) * cos(theta2 - gamma)
        cos_inc = np.cos(theta1) * cos_rel_az
        
        solar_flux_on_wall = I0 * max(0, cos_inc)
        
        # Illuminance
        # E_in = I_vis0 * cos(theta1) * cos(theta2 - gamma) * ...
        I_vis0 = I0 * EFFICACY
        I_vis_wall = I_vis0 * max(0, cos_inc)
        
        E_in = I_vis_wall * (self.cfg.window_area / self.cfg.floor_area) * self.cfg.tau_visible * (1.0 - f_shade) * self.cfg.c_room_light
        
        # 覆盖旧的 solar_flux_base 以获得更准确的热计算
        solar_flux_base = solar_flux_on_wall

        # PDE 墙体导热
        q_rad_wall = self.cfg.k_const_absorb * solar_flux_base

        flux_out = self.cfg.h_out * (T_out + q_rad_wall - self.T_wall[0])
        flux_in = self.cfg.h_in * (self.T_wall[self.N_nodes] - self.T_in)
        
        Fo = self.alpha * self.dt / (self.dx**2)
        T_w_new = self.T_wall.copy()
        
        # 内部节点
        T_w_new[1:-1] = self.T_wall[1:-1] + Fo * (self.T_wall[2:] - 2*self.T_wall[1:-1] + self.T_wall[:-2])
        
        # 边界节点
        T_w_new[0] = self.T_wall[0] + Fo * (2*self.T_wall[1] - 2*self.T_wall[0] + 2*self.dx * flux_out / self.cfg.k_wall)
        T_w_new[-1] = self.T_wall[-1] + Fo * (2*self.T_wall[-2] - 2*self.T_wall[-1] - 2*self.dx * flux_in / self.cfg.k_wall)
        
        self.T_wall = T_w_new
        
        # ODE 室内温度
        # Q_solar_gain (Through Window)
        Q_sol = solar_flux_base * self.cfg.window_area * self.cfg.tau_window * (1.0 - f_shade)
        Q_win = self.cfg.u_window * self.cfg.window_area * (T_out - self.T_in)
        Q_wall = self.cfg.h_in * self.cfg.solid_wall_area * (self.T_wall[-1] - self.T_in)
        
        # Q_ventilation (New)
        # ACH * V * rho_air * c_air * (T_out - T_in) / 3600
        rho_air = 1.225
        c_air = 1005.0
        
        # Dynamic Ventilation Logic (Night Flushing)
        current_ach = self.cfg.ventilation_rate
        if self.cfg.night_cooling:
             # Use clock time for simplicity. Night: 8PM (20:00) to 7AM (7:00)
             # Only ventilate if outside is cooler than inside
             h = t_current.hour
             if (h >= 20 or h < 7) and T_out < self.T_in:
                 current_ach = self.cfg.night_vent_rate

        # ACH -> Hz: x / 3600
        m_dot_cp = (current_ach * self.cfg.room_volume * rho_air * c_air) / 3600.0
        Q_vent = m_dot_cp * (T_out - self.T_in)
        
        dQ = self.cfg.Q_internal + Q_sol + Q_win + Q_wall + Q_vent
        dT_in = (dQ / self.C_in) * self.dt
        
        self.T_in += dT_in
        
        return {
            'time': t_current,
            'T_out': T_out,
            'T_in': self.T_in,
            'T_wall_outer': self.T_wall[0],
            'T_wall_inner': self.T_wall[-1],
            'Solar_Flux': solar_flux_base,
            'Illuminance': E_in,
            'F_shade': f_shade,
            'Q_sol_gain': Q_sol,
            'Q_wall_gain': Q_wall,
            'Q_vent_loss': Q_vent,
            'Q_internal': self.cfg.Q_internal,
            'Q_heating_load': max(0, 20.0 - self.T_in) * self.C_in / 3600.0 if self.T_in < 20 else 0,
            'Q_cooling_load': max(0, self.T_in - 26.0) * self.C_in / 3600.0 if self.T_in > 26 else 0
        }

    def simulate(self, weather_df):
        results = []
        print(f"Starting simulation for {len(weather_df)} steps...")
        
        # 假设 weather_df index 是 datetime 或者有 time 列
        if 'time' in weather_df.columns:
             # 处理 PVGIS 时间格式 yyyymmdd:HHMM
             if isinstance(weather_df['time'].iloc[0], str):
                 times = pd.to_datetime(weather_df['time'], format='%Y%m%d:%H%M')
             else:
                 times = weather_df['time']
        else:
             # creating dummy times if needed, or index
             times = pd.date_range(start='2023-01-01', periods=len(weather_df), freq='H')

        for idx in range(len(weather_df)):
            t_curr = times[idx]
            row = weather_df.iloc[idx]
            
            # 一个小时的数据，但我们要跑很多个 dt 步长
            steps_per_hour = int(3600 / self.dt)
            
            res_snapshot = None
            for _ in range(steps_per_hour):
                 res_snapshot = self.step(t_curr, row)
            
            results.append(res_snapshot)
            
        return pd.DataFrame(results)