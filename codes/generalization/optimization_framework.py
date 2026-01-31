import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import sys
import os

# 确保可以导入 calculate.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from calculate import BuildingConfig, ThermalSystem, OverhangStats, VerticalFins, NoShading, SolarModel

# 设置绘图风格
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

@dataclass
class OptimizationWeights:
    """定义优化目标的权重配置"""
    # 权重系数 (Weights)
    w_comfort_dev: float = 1.0     # 舒适度偏差权重 (单位：摄氏度)
    w_temp_var: float = 0.5        # 温度波动性权重
    w_energy: float = 0.0001       # 能耗权重 (注意量级差异, Q通常很大)
    w_cost: float = 0.01           # 初始造成本权重
    w_light_quality: float = 0.1   # 光照质量权重 (负值代表越多越好，正值代表惩罚)

    # 目标设定
    target_temp_min: float = 20.0
    target_temp_max: float = 26.0
    
    # 归一化因子 (Scale Factors) - 用于将不同量纲的物理量映射到同一水平
    scale_temp: float = 1.0        # 1 degC
    scale_energy: float = 1e6      # 1 MJ
    scale_cost: float = 1000.0     # 1000 USD/CNY
    scale_light: float = 1e5       # 100 kW flux

class MetricEvaluator:
    def __init__(self, weights: OptimizationWeights):
        self.w = weights

    def evaluate(self, time_series_df: pd.DataFrame, strategy_cost: float):
        """
        根据仿真结果的时间序列计算综合评分。
        :param time_series_df: 包含 T_in, Q_cooling_load, Q_sol_gain 等列的 DataFrame
        :param strategy_cost: 该策略的预估建设成本
        :return: 包含总分及各项子分的字典
        """
        # 1. 舒适度指标 (Comfort)
        # 计算偏离舒适区间的程度 (Degree Hours of Discomfort)
        t_in = time_series_df['T_in'].values
        discomfort = np.zeros_like(t_in)
        
        # 低于最小值
        mask_low = t_in < self.w.target_temp_min
        discomfort[mask_low] = self.w.target_temp_min - t_in[mask_low]
        
        # 高于最大值
        mask_high = t_in > self.w.target_temp_max
        discomfort[mask_high] = t_in[mask_high] - self.w.target_temp_max
        
        avg_discomfort = np.mean(discomfort) # 平均偏差度数
        
        # 2. 稳定性指标 (Variance)
        temp_std = np.std(t_in)
        
        # 3. 能耗指标 (Energy)
        # 累加制冷和制热负载
        total_energy = time_series_df['Q_cooling_load'].sum() + time_series_df['Q_heating_load'].sum()
        
        # 4. 光照质量指标 (Light Quality)
        # 简易代理：获得的太阳辐射总量 (Q_sol_gain)
        # 假设：适量的光是好的，但该指标更关注是否"过度"导致能耗，或者是否"过少"导致需要人工照明
        # 对于Sungrove (热带)，我们希望尽量减少得热，但保留采光。
        # 这里简化为：总辐射量 (作为一种环境影响的度量)
        total_solar_gain = time_series_df['Q_sol_gain'].sum()
        
        # 计算归一化得分 (Score, 越低越好)
        # Score = Cost function to minimize
        
        score = (
            self.w.w_comfort_dev * (avg_discomfort / self.w.scale_temp) +
            self.w.w_temp_var * (temp_std / self.w.scale_temp) +
            self.w.w_energy * (total_energy / self.w.scale_energy) +
            self.w.w_cost * (strategy_cost / self.w.scale_cost) - 
            self.w.w_light_quality * (total_solar_gain / self.w.scale_light) # 减去光照收益? 或者加上光照过载惩罚?
            # 修改逻辑：光照作为收益项不好量化，这里通常光照多 -> 热量多 -> 能耗高。
            # 所以光照本身已经在能耗里体现了负面。
            # 只有当需要"自然采光"时，光照才是收益。
            # 让我们假设 target 是维持一定的光通量。
            # 暂时简化：不计入独立的光照分，因为它与热强耦合，直接看能耗和舒适度即可。
        )
        
        return {
            "Total_Score": score,
            "Discomfort_Avg": avg_discomfort,
            "Temp_Std": temp_std,
            "Total_Energy_MJ": total_energy / 1e6, # 转换为MJ方便阅读
            "Strategy_Cost": strategy_cost,
            "Raw_Total_Solar_MJ": total_solar_gain / 1e6
        }

class ParameterOptimizer:
    def __init__(self, base_config: BuildingConfig, weather_df: pd.DataFrame, location_lat: float, weights: OptimizationWeights):
        self.base_config = base_config
        self.weather_df = weather_df
        self.location_lat = location_lat
        self.evaluator = MetricEvaluator(weights)
        
    def estimate_cost(self, param_name, param_value, facade_area):
        """简单的成本估算模型"""
        # 假设成本与遮阳装置的尺寸成正比
        unit_price_per_m2_material = 200.0 # 200元/平米材料费
        
        if param_name == 'overhang_depth':
            # 遮阳板面积 = 宽度(W) * 深度(L)
            # 假设窗户宽度占墙面宽度的比例与WWR有关，这里做简化估算
            # 假设每平米窗户上方有一个深度为 param_value 的板
            # Cost ~ Total Window Width * Depth * Price
            # Window Area = Facade Area * WWR
            # Window Width approx sqrt(Area) or similar? 
            # 简化：Cost = 遮阳板深度 * 窗户总宽 * 单价
            # 设窗户总宽 = 墙面积 * 窗墙比 (这是一个非常粗略的几何假设，仅用于演示逻辑)
            effective_width = facade_area * self.base_config.window_ratio
            return effective_width * param_value * unit_price_per_m2_material
            
        elif param_name == 'fin_depth':
            # 垂直遮阳板
            # Cost ~ Height * Depth * Count
            effective_area = facade_area * self.base_config.window_ratio
            return effective_area * param_value * unit_price_per_m2_material
            
        elif param_name == 'window_ratio':
            # 窗墙比改变也会改变成本（玻璃通常比墙贵）
            glass_price = 300.0
            wall_price = 100.0
            total_cost = (facade_area * param_value * glass_price) + (facade_area * (1-param_value) * wall_price)
            # 我们只关心相对于基准的变化成本，或者绝对成本
            return total_cost
            
        return 0.0

    def run_sweep(self, param_name: str, param_range: list, shading_type='Overhang', facade_azimuth=0):
        """
        对单个参数进行扫描
        """
        results = []
        
        print(f"开始扫描参数: {param_name}, 范围: {param_range}, 类型: {shading_type}")
        
        for val in param_range:
            # 1. 配置修改
            # 使用 dataclasses.replace 创建副本，避免污染原始配置
            from dataclasses import replace
            if hasattr(self.base_config, param_name):
                # 如果是 config 里的参数 (e.g. window_ratio)
                current_config = replace(self.base_config, **{param_name: val})
            else:
                # 如果是外部参数 (e.g. overhang_depth)，config 保持不变
                current_config = self.base_config
                
            # 2. 遮阳策略生成
            # 假设标准窗户尺寸 2m x 1.5m
            win_h, win_w = 2.0, 1.5 
            
            if shading_type == 'Overhang':
                depth = val if param_name == 'overhang_depth' else 0.5
                shade_strategy = OverhangStats(win_h, win_w, facade_azimuth, depth_L=depth)
            elif shading_type == 'Fins':
                depth = val if param_name == 'fin_depth' else 0.5
                shade_strategy = VerticalFins(win_h, win_w, facade_azimuth, depth_L=depth)
            else:
                shade_strategy = NoShading(win_h, win_w, facade_azimuth)
                
            # 3. 运行模拟
            system = ThermalSystem(current_config, shade_strategy, self.location_lat)
            # 运行部分数据以加快速度 (比如只跑最热的一个月)
            # 为了演示，我们只取前 720 小时 (30天)
            subset_weather = self.weather_df.iloc[0:720].copy() 
            res_df = system.simulate(subset_weather)
            
            # 4. 评估指标
            cost = self.estimate_cost(param_name, val if param_name in ['overhang_depth', 'fin_depth', 'window_ratio'] else 0, 420.0)
            metrics = self.evaluator.evaluate(res_df, cost)
            
            # 记录结果
            metrics[param_name] = val
            results.append(metrics)
            
        return pd.DataFrame(results)

def plot_optimization_results(df_results, x_col, title):
    """
    绘制多轴图表：展示综合得分以及各子项的变化趋势
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel(x_col)
    ax1.set_ylabel('Total Score (Lower is Better)', color=color)
    ax1.plot(df_results[x_col], df_results['Total_Score'], color=color, marker='o', label='Total Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    # 双轴显示能耗
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Total Energy (MJ)', color=color)  
    ax2.plot(df_results[x_col], df_results['Total_Energy_MJ'], color=color, linestyle='--', marker='x', label='Energy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(title)
    fig.tight_layout()
    plt.show()

# ==========================================
# 示例运行逻辑
# ==========================================
if __name__ == "__main__":
    # 模拟数据加载 (用户可以替换为真实load_weather函数)
    # 这里创建一个假的 weather_df 供演示
    dates = pd.date_range('2023-01-01', periods=720, freq='H')
    weather_mock = pd.DataFrame({
        'time': dates,
        'T2m': 25 + 5 * np.sin(np.linspace(0, 60, 720)), # 20-30度波动
        'Gb(i)': np.maximum(0, 800 * np.sin(np.linspace(0, 60, 720) - 0.5)), # 白天有光
    })

    # 配置权重 (用户可以根据现实情况调整这些超参数)
    weights = OptimizationWeights(
        w_comfort_dev=2.0,   # 非常在意舒适度
        w_energy=0.005,      # 关注节能
        w_cost=0.5           # 稍微关注成本
    )
    
    base_cfg = BuildingConfig(
        C_in=800000.0, Q_internal=200.0, layer_thickness=0.3, 
        wall_area=1.0, window_ratio=0.3, k_wall=0.8, 
        rho_wall=1800.0, c_wall=1000.0, h_in=8.0, h_out=20.0, 
        u_window=2.5, tau_window=0.7
    )
    
    optimizer = ParameterOptimizer(base_cfg, weather_mock, location_lat=1.35, weights=weights)
    
    # 扫描遮阳板深度
    print("\n--- Running Optimization Sweep for Overhang Depth ---")
    depths = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    results = optimizer.run_sweep('overhang_depth', depths, shading_type='Overhang', facade_azimuth=0) # South
    
    print("\nOptimal Parameters found:")
    best_row = results.loc[results['Total_Score'].idxmin()]
    print(best_row)
    
    plot_optimization_results(results, 'overhang_depth', 'Impact of Overhang Depth on Performance Score')
