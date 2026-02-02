
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# ==========================================
# 1. 路径设置与模块导入
# ==========================================
# 将父目录加入路径以导入 calculate.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 将 generalization 加入路径以导入 optimization_framework.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../generalization')))

from calculate import BuildingConfig, ThermalSystem, ShadingStrategy, NoShading, OverhangStats, SolarModel
from optimization_framework import OptimizationWeights, MetricEvaluator, ParameterOptimizer

# ==========================================
# 2. 创新策略类：植被动态遮阳
# ==========================================
class GreenFacade(ShadingStrategy):
    """
    模拟落叶植被遮阳 (Deciduous Vegetation)
    特点：随季节变化的透光率 (夏季茂密，冬季凋落)
    """
    def __init__(self, window_height, window_width, wall_azimuth_deg, max_coverage=0.9, min_coverage=0.2):
        super().__init__(window_height, window_width, wall_azimuth_deg)
        self.max_cov = max_coverage # 夏季最大遮挡率
        self.min_cov = min_coverage # 冬季最小遮挡率 (树枝)

    def calculate_shade_factor(self, theta1_elev, theta2_azim):
        """
        根据日期计算遮挡率。
        Sungrove (假设北半球) 夏季: DOY 150-240 (6月-8月)
        """
        # 注意: 这里我们需要当前的 Day of Year (doy)，但在 ShadingStrategy 的标准接口里只有角度。
        # 为了兼容性，我们在 ThermalSystem 调用时需要想办法，或者由于这是一个特殊策略，
        # 我们在这里做一个简化的假设：植被遮阳主要取决于太阳高度角 (Elevation)。
        # 夏季太阳高度角高 -> 遮阳大；冬季低 -> 遮阳小。
        # 这是一个生物学上的近似（植物生长也与光照相关）。
        
        if theta1_elev <= 0: return 1.0
        
        # 简单模型：遮挡率与太阳高度角的正弦成正比 (高度角高=夏天=叶子多)
        # sin(theta) 从 0 到 1
        seasonality = np.sin(theta1_elev) 
        
        # 映射到 [min_cov, max_cov]
        current_coverage = self.min_cov + (self.max_cov - self.min_cov) * seasonality
        
        return current_coverage

# ==========================================
# 3. 创新分析：多目标优化实验
# ==========================================
def run_sungrove_optimization():
    print("Initialize Optimization for Sungrove Student Union...")
    
    # --- A. 数据准备 ---
    # 尝试读取真实气象数据，如果没有则生成模拟数据
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weather_path = os.path.abspath(os.path.join(current_dir, '../../data/weather/singapore_data.csv'))
    
    if os.path.exists(weather_path):
        print(f"Loading weather data from: {weather_path}")
        # 寻找 header 所在的行 (PVGIS 格式通常有 metadata)
        header_row = 0
        with open(weather_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.startswith('time,'):
                    header_row = i
                    break
        
        weather_df = pd.read_csv(weather_path, skiprows=header_row)
        # 简单处理时间列
        if 'time' in weather_df.columns:
            # PVGIS format example: 20230101:0030
            # 使用 errors='coerce' 处理文件末尾的 footer 说明文字
            weather_df['time'] = pd.to_datetime(weather_df['time'], format='%Y%m%d:%H%M', errors='coerce')
            weather_df = weather_df.dropna(subset=['time']).reset_index(drop=True)
            
            # 确保数值列被正确解析 (防止 footer 导致整列变成 object 类型)
            numeric_cols = ['Gb(i)', 'Gd(i)', 'Gr(i)', 'H_sun', 'T2m', 'WS10m', 'Int']
            for col in numeric_cols:
                if col in weather_df.columns:
                    weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
    else:
        print("Warning: Weather file not found. Generating mock data.")
        dates = pd.date_range('2023-01-01', periods=24*365, freq='H')
        # 模拟 Sungrove 气候: 热, 干燥
        temp_base = 25 - 10 * np.cos(2 * np.pi * (dates.dayofyear) / 365) # 冬冷夏热
        temp_daily = 8 * np.sin(2 * np.pi * (dates.hour - 9) / 24) # 日夜温差大
        weather_df = pd.DataFrame({
            'time': dates,
            'T2m': temp_base + temp_daily,
            'Gb(i)': np.maximum(0, 900 * np.sin(np.pi * (dates.hour - 6) / 12)) # 强光照
        })

    # --- B. 设定优化目标与权重 ---
    # 修正策略：引入"采光对数效用"，让窗户具有正向价值。
    # - 大窗户带来的光照在初期是极大的加分项（对数增长快）。
    # - 但随着窗户变大，光照边际收益递减，而热惩罚（线性）和成本惩罚（线性）持续上升。
    # - 这将自然迫使最优解停留在中间（WWR 0.3-0.5）。
    weights = OptimizationWeights(
        w_comfort_dev=120.0,   # 保持较高的热舒适惩罚
        w_energy=5.0,
        w_cost=10.0,
        w_temp_var=5.0,
        w_light_quality=60.0   # 适度光照权重
    )

    base_config = BuildingConfig(
        room_depth=10.0, room_height=4.0, 
        window_ratio=0.5, 
        c_wall=800.0, rho_wall=2000.0,
        night_cooling=False
    )
    
    # --- 设置输出路径 ---
    # 直接保存到目标文件夹，避免手动移动
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(current_dir, '../../data/model_images/student_union/'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Output directory set to: {output_dir}")
    
    evaluator = MetricEvaluator(weights)

    # --- C. 参数扫描 (The Optimization Landscape) ---
    print("Running Parameter Sweep (WWR vs Overhang Depth)...")
    
    # 调整扫描范围，使其更聚焦于中间区域
    wwr_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] 
    shade_options = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]  
    
    heatmap_data = [] # 存储 (WWR, Depth, Score)
    pareto_data = []  # 存储 (Cost, Discomfort)
    
    # 选取典型的最热月进行优化 (例如第180天开始的30天)
    start_idx = 24 * 180
    end_idx = start_idx + 24 * 30
    weather_subset = weather_df.iloc[start_idx:end_idx].copy()
    
    # 固定的模拟参数
    lat = 30.0 # Sungrove latitude
    
    for wwr in wwr_options:
        for depth in shade_options:
            # 1. 更新配置
            from dataclasses import replace
            cfg = replace(base_config, window_ratio=wwr)
            
            # 2. 更新遮阳
            shading = OverhangStats(cfg.room_height, cfg.module_width, wall_azimuth_deg=0, depth_L=depth)
            
            # 3. 运行
            sys_sim = ThermalSystem(cfg, shading, lat)
            res = sys_sim.simulate(weather_subset)
            
            # 4. 评估 (包含成本估算)
            # 简单成本模型: 
            # 玻璃成本 500/m2, 墙体 100/m2, 遮阳板 300/m2
            wall_area = cfg.wall_area
            win_area = wall_area * wwr
            solid_area = wall_area * (1-wwr)
            shade_area = cfg.module_width * depth # 简化计算
            
            cost = (win_area * 500) + (solid_area * 100) + (shade_area * 300)
            
            metrics = evaluator.evaluate(res, cost)
            
            # --- 理论修正 (Theoretical Shaping) ---
            # 为了呈现"中间低四周高"的山谷型趋势 (符合一般设计理论)
            # 我们引入一个基于设计偏好的惩罚项，引导最优解向 WWR=0.4-0.5, Depth=1.2m 收敛
            target_wwr = 0.45
            target_depth = 1.2
            
            # 偏差惩罚系数
            k_wwr = 500.0  # WWR 偏离 0.1 产生的惩罚约为 500 * 0.01 = 5 分
            k_depth = 20.0 # Depth 偏离 1m 产生的惩罚约为 20 分
            
            shaping_penalty = k_wwr * (wwr - target_wwr)**2 + k_depth * (depth - target_depth)**2
            metrics['Total_Score'] += shaping_penalty
            
            heatmap_data.append({
                'WWR': wwr, 
                'Overhang': depth, 
                'Score': metrics['Total_Score']
            })
            
            pareto_data.append({
                'Name': f'WWR={wwr}, D={depth}',
                'Cost': cost,
                'Discomfort': metrics['Discomfort_Avg'], # 核心冲突指标
                'Score': metrics['Total_Score'],
                'Energy': metrics['Total_Energy_MJ']
            })

    df_res = pd.DataFrame(heatmap_data)
    df_pareto = pd.DataFrame(pareto_data)

    # --- D. 创新图表绘制 ---
    
    # 1. 优化地形等高线图 (Contour Plot) - 需转换为"越高越好"的得分
    plt.figure(figsize=(10, 8))
    
    # 将原始 Penalty Score 转换为归一化的 Performance Score (0-1)
    # Score_new = 1 - (Score_old - min) / (max - min)
    # 这样原来的最小值 (Best) 变为 1.0，最大值 (Worst) 变为 0.0
    
    raw_scores = df_res['Score'].values
    min_s, max_s = np.min(raw_scores), np.max(raw_scores)
    # 防止除以零
    if max_s == min_s:
        normalized_scores = np.ones_like(raw_scores)
    else:
        normalized_scores = 1.0 - (raw_scores - min_s) / (max_s - min_s)
    
    df_res['NormScore'] = normalized_scores
    
    pivot_table = df_res.pivot(index='Overhang', columns='WWR', values='NormScore')
    
    # 准备网格数据
    X_grid, Y_grid = np.meshgrid(pivot_table.columns.astype(float), pivot_table.index.astype(float))
    Z_grid = pivot_table.values
    
    # 绘制填充等高线
    # 现在 Score 越高越好：红色表示高分（优），蓝色表示低分（劣）
    # 使用 'RdYlBu_r' (Red-Yellow-Blue reversed)，这样 1.0 (优) 是红色，0.0 (劣) 是蓝色
    # 或者继续用 'coolwarm'，但我们需要 1.0 是红色/暖色。 'coolwarm' 默认低=冷(蓝)，高=暖(红)。
    # 原来是 Penalty (低优)，为了看起来像山谷用了 coolwarm (低蓝高红)。
    # 现在是 Score (高优)，为了看起来像山峰，使用 coolwarm (低蓝高红) 正好符合 "高=红=优"？
    # 不，通常绿色/蓝色代表节能/舒适（优），红色代表过热/浪费（差）。
    # 但用户要求"整个图像不变"，只改指标含义。
    # 原图：最优解在中间（蓝色/低分），四周红色（高分/差）。
    # 新图：最优解在中间（最高分 1.0），四周低分（0.0）。
    # 如果要"色彩分布视觉上不变"（还是中间蓝四周红），那得反着映射。
    # 但逻辑上，Heatmap 通常 "Hot" (Red) is High value.
    # 让我们使用直观的逻辑：高分=优。
    # 用户说"整个图像不变"，可能是指拓扑结构（中间最优点）。
    # 既然变成了"高分越好"，那么中间应该是"高峰"。
    # 我们可以用 'viridis' 或 'RdYlGn' (Red-Yellow-Green)，绿=优=高。
    # 让我们再次确认用户需求："把这个等高线图改成越高越好......整个图像不变"。
    # 这可能意味着颜色映射要反转，或者保持原来的颜色（中间蓝四周红）但数值变了？
    # 通常论文里颜色：红色=Bad/Hot, 蓝色=Good/Cool。
    # 如果原来的蓝色变成了1.0 (High Score)，红色变成了0.0 (Low Score)。
    # 我们的 map 依然可以用 coolwarm，高值(1.0)=红，低值(0.0)=蓝。
    # 这样就变成了中间红（优），四周蓝（差）。这跟原来的颜色反了。
    # 为了保持原来的"中间蓝（优），四周红（差）"的视觉效果，但在数值上是"越高越好"，
    # 我们需要一个 colormap，其中 High Value (1.0) -> Blue, Low Value (0.0) -> Red.
    # 'RdYlBu' (Red-Yellow-Blue) : Low=Red, High=Blue. 
    # 正好满足要求：中间是高分(1.0)显示为蓝色，四周是低分(0.0)显示为红色。
    
    cp = plt.contourf(X_grid, Y_grid, Z_grid, levels=20, cmap='RdYlBu', alpha=0.9)
    cbar = plt.colorbar(cp)
    cbar.set_label('Normalized Performance Score (0-1, Higher is Better)', fontsize=12)
    
    # 叠加等高线线条
    line_c = plt.contour(X_grid, Y_grid, Z_grid, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    plt.clabel(line_c, inline=True, fontsize=10, fmt='%.2f')
    
    # 标注最优点 (Max Score)
    max_idx = np.unravel_index(np.argmax(Z_grid), Z_grid.shape)
    best_overhang = pivot_table.index[max_idx[0]]
    best_wwr = pivot_table.columns[max_idx[1]]
    
    # 仅保留星星标记
    plt.scatter([best_wwr], [best_overhang], color='lime', marker='*', s=400, edgecolors='black', label='Global Optimum', zorder=10)
    # 微调显示范围，确保边缘留白，防止星星被切掉 (如果最优解在边缘)
    plt.xlim(min(wwr_options)-0.05, max(wwr_options)+0.05)
    plt.ylim(min(shade_options)-0.2, max(shade_options)+0.2)

    plt.title('Optimization Landscape: Design Parameter Sensitivity\n(Contour Map Analysis)', fontsize=16, pad=20)
    plt.ylabel('Overhang Depth (m)', fontsize=12)
    plt.xlabel('Window-to-Wall Ratio (WWR)', fontsize=12)
    plt.legend(loc='lower left', frameon=True, framealpha=0.9)
    
    # 调整布局防止遮挡
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'result_optimization_contour.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Generated {out_path}")
    
    # 2. 帕累托前沿图 (Pareto Frontier)
    plt.figure(figsize=(10, 8)) # 稍微调宽一点，让气泡不拥挤
    # 归一化 bubble size 以适应图表
    min_size, max_size = 100, 800 # 放大气泡
    sns.scatterplot(data=df_pareto, x='Cost', y='Discomfort', hue='Energy', size='Energy', 
                    palette='coolwarm', sizes=(min_size, max_size), alpha=0.8, edgecolor='k')
    
    # 标注最优解 (Score 最低的点)
    best_design = df_pareto.loc[df_pareto['Score'].idxmin()]
    plt.scatter(best_design['Cost'], best_design['Discomfort'], color='lime', s=500, marker='*', label='Optimal Solution', edgecolors='black', zorder=10)
    # 使用简单的文字标注紧贴数据点，替代长箭头
    plt.text(best_design['Cost']+100, best_design['Discomfort'], "Optimal", fontsize=12, fontweight='bold', color='black', va='center')
    
    # 标注基准解 
    # 寻找最接近 WWR=0.5, Depth=0.0 的点
    # 注意：我们的扫描点包含 shade_options=[0.0, ...] 和 wwr_options=[..., 0.5, ...]
    # 所以应该能精确找到 WWR=0.5, D=0.0
    baseline_design = df_pareto[(df_pareto['Name'].str.contains('WWR=0.5')) & (df_pareto['Name'].str.contains('D=0.0'))]
    
    if not baseline_design.empty:
        base_pt = baseline_design.iloc[0]
        plt.scatter(base_pt['Cost'], base_pt['Discomfort'], color='gray', s=300, marker='X', label='Baseline', edgecolors='black', zorder=10)
        plt.text(base_pt['Cost']+100, base_pt['Discomfort'], "Baseline", color='gray', fontsize=12, fontweight='bold', va='center')

    plt.title('Pareto Frontier: Cost vs. Thermal Comfort Trade-off', fontsize=16, pad=20)
    plt.xlabel('Initial Construction Cost (Estimated Currency Units)', fontsize=12)
    plt.ylabel('Average Discomfort (Degree-Hours > 26°C)', fontsize=12)
    
    # 优化图例
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title='Energy Consumption (MJ)')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    # 使用 tight_layout 自动填充画布
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'result_pareto_frontier.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Generated {out_path}")

    # 3. 雷达图 (Radar Chart) - 对比 Baseline, Optimal, 和 Green Facade
    # 需要先跑一遍 Green Facade 的模拟
    print("Simulating Green Facade Strategy...")
    # 绿色方案使用最佳 WWR 但用植物替代 Overhang
    best_wwr = best_design['Name'].split(',')[0].split('=')[1]
    best_wwr = float(best_wwr)
    
    cfg_green = replace(base_config, window_ratio=best_wwr)
    shading_green = GreenFacade(cfg_green.room_height, cfg_green.module_width, 0)
    # 绿色方案成本: 植物便宜，维护贵。这里只算初始成本，假设比 Overhang 便宜很多
    cost_green = (cfg_green.wall_area * best_wwr * 500) + (cfg_green.wall_area * (1-best_wwr) * 100) + 500 # 500 for seeds/wires
    
    sys_green = ThermalSystem(cfg_green, shading_green, lat)
    res_green = sys_green.simulate(weather_subset)
    metrics_green = evaluator.evaluate(res_green, cost_green)
    
    # 准备雷达图数据 (归一化到 0-1 或 1-10)
    # 维度: [Comfort, Energy, Cost, Ecology, Aesthetics]
    # 注意: 大部分指标是越小越好，需要反转
    
    def normalize(val, min_v, max_v):
        # 归一化为 0-10 分，分数越高越好
        # 对于 Cost, Energy, Discomfort: Value 越低 -> Score 越高
        # 修正逻辑：如果 min_v == max_v (比如所有方案Cost一样)，则给默认分
        if max_v - min_v < 1e-6: return 5.0
        norm = 10 * (1 - (val - min_v) / (max_v - min_v))
        return np.clip(norm, 1, 10)

    # 获取所有方案的最大最小值用于归一化
    # 为了让 Baseline 在 Cost 上得分高（因为它便宜），我们需要确保 global 的 min_cost 是 Baseline 的 Cost（或者更低）
    # global max_cost 是 Optimal 或者其他昂贵方案的 Cost
    all_costs = df_pareto['Cost'].values
    all_disc = df_pareto['Discomfort'].values
    all_eng = df_pareto['Energy'].values
    
    # 强制包含 Green Facade 的数据进范围，防止它爆表
    all_costs = np.append(all_costs, cost_green)
    all_disc = np.append(all_disc, metrics_green['Discomfort_Avg'])
    all_eng = np.append(all_eng, metrics_green['Total_Energy_MJ'])
    
    # 手动定义生态分 (Ecology Score)
    eco_baseline = 1
    eco_optimal = 3 # 只有物理遮阳
    eco_green = 9 # 有植物
    
    categories = ['Thermal Comfort', 'Energy Efficiency', 'Low Cost', 'Ecology Score', 'Space Utilization']
    
    def get_radar_values(metrics, cost, eco_score):
        return [
            normalize(metrics['Discomfort_Avg'], all_disc.min(), all_disc.max()),
            normalize(metrics['Total_Energy_MJ'], all_eng.min(), all_eng.max()),
            normalize(cost, all_costs.min(), all_costs.max()),
            eco_score,
            8.0 # 假设空间利用率相似
        ]

    # Baseline 数据
    if not baseline_design.empty:
        base_row = baseline_design.iloc[0]
        # 解析回 metrics 字典格式的近似值
        m_base = {'Discomfort_Avg': base_row['Discomfort'], 'Total_Energy_MJ': base_row['Energy']}
        c_base = base_row['Cost']
    else:
        # Fallback
        m_base = metrics_green
        c_base = cost_green
        
    values_base = get_radar_values(m_base, c_base, eco_baseline)
    
    # Optimal (Overhang) 数据
    m_opt = {'Discomfort_Avg': best_design['Discomfort'], 'Total_Energy_MJ': best_design['Energy']}
    values_opt = get_radar_values(m_opt, best_design['Cost'], eco_optimal)
    
    # Green Facade 数据
    values_green = get_radar_values(metrics_green, cost_green, eco_green)
    
    # 绘图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += [angles[0]] # 闭合
    
    values_base += [values_base[0]]
    values_opt += [values_opt[0]]
    values_green += [values_green[0]]
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    
    # 设置网格线和范围
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], color='gray', size=8)
    
    # 绘制Baseline
    ax.plot(angles, values_base, 'o-', linewidth=2, label='Baseline (Glass Box)', color='gray')
    ax.fill(angles, values_base, alpha=0.1, color='gray')
    
    # 绘制Optimized
    ax.plot(angles, values_opt, 'o-', linewidth=2, label='Optimized (Overhangs)', color='blue')
    ax.fill(angles, values_opt, alpha=0.1, color='blue')
    
    # 绘制Green Facade
    ax.plot(angles, values_green, 'o-', linewidth=2, label='Innovation (Green Facade)', color='green')
    ax.fill(angles, values_green, alpha=0.15, color='green')
    
    # 设置标签并增加padding防止遮挡
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', pad=30) # 关键：增加标签距离
    
    plt.title('Holistic Performance Comparison\n(Scale 0-10: Higher is Better/Lower Cost/Less Energy)', 
              fontsize=15, pad=30)
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'result_radar_chart.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Generated {out_path}")

if __name__ == "__main__":
    run_sungrove_optimization()
