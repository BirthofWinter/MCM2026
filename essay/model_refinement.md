# 模型深化与扩展说明 (Model Refinement & Extension)

为了实现比基础遮阳更高的节能目标，并解决跨气候区（热带/寒带）的特定热工问题，我们将基础物理模型扩展为**“动态耦合热系统”**。该部分的深化内容包括主动通风逻辑、表面辐射特性优化以及非稳态热惯性效应。

## 1. 针对桑格罗夫（Sungrove）的增强型热模型
*Deep Retrofit Model for Tropical Climates*

在热带气候下，为了最大化被动式节能（目标节能率 > 40%），模型引入了以下子系统：

### 1.1 智能夜间通风控制方程 (Smart Night Flushing)
在传统稳态计算中，通风率 $ACH$ (Air Changes per Hour) 通常被视为常数。我们在模型中引入了**温差驱动的动态控制逻辑**。当检测到室外温度低于室内温度且处于夜间非工作时间时，系统模拟自动开启高风量通风，利用夜间冷空气带走白天蓄积在墙体和家具中的热量。

改进后的通风散热量 $Q_{vent}(t)$ 表达为：

$$ Q_{vent}(t) = \frac{V \cdot \rho_{air} \cdot c_{air} \cdot ACH(t)}{3600} \cdot (T_{out}(t) - T_{in}(t)) $$

其中，$ACH(t)$ 是一个分段控制函数：

$$
ACH(t) = 
\begin{cases} 
ACH_{night} (8.0), & \text{if } (t \in \text{Night}) \land (T_{out} < T_{in}) \\
ACH_{base} (2.0), & \text{otherwise}
\end{cases}
$$

*   **物理意义**：利用凉爽的夜间空气作为天然冷源 (Natural Heat Sink)，减少第二天白天开启空调前的初始热负荷。
*   **模型参数**：在模拟中设定 $ACH_{night}=8.0$，模拟强制排风或全开窗效果。

### 1.2 冷包围护结构 (Cool Envelope & Albedo)
为了降低太阳辐射对不透明墙体的热冲击，模型修正了外墙边界条件中的**综合温度（Sol-air Temperature）**项。我们引入了表面吸收率参数 $\alpha$（对应代码中的 `k_const_absorb`）。

墙体外表面 ($x=0$) 的能量平衡方程修正为：

$$ -k \frac{\partial T}{\partial x}\bigg|_{x=0} = h_{out} \left( T_{out} + \underbrace{\frac{\alpha \cdot I_{solar}}{h_{out}}}_{\text{Equivalent Solar Temp}} - T_{wall}(0,t) \right) $$

*   **参数调整**：
    *   **Baseline**: $\alpha = 0.6$ (普通混凝土/砖墙)
    *   **Deep Retrofit**: $\alpha = 0.3$ (高反射白色涂料/冷屋顶材料)
*   **物理意义**：降低 $\alpha$ 直接减少了进入墙体传导层的净热通量，模拟了“冷屋顶”技术。

### 1.3 高性能玻璃模型 (High-Performance Glazing)
窗户的热增益由两部分组成：直接透射辐射和温差传导。改进模型采用了 **Low-E（低辐射）双层中空玻璃** 的参数。

$$ Q_{window} = \underbrace{A_{win} \cdot U_{value} \cdot (T_{out} - T_{in})}_{\text{Conduction}} + \underbrace{A_{win} \cdot SHGC \cdot I_{solar} \cdot (1 - F_{shade})}_{\text{Radiation}} $$

*   **参数优化**：
    *   $U_{value}$：从 $2.5 W/m^2K$ (普通单层) 降至 $1.5 W/m^2K$ (Low-E双层)。
    *   $SHGC$ ($\tau$)：从 $0.7$ 降至 $0.4$，这意味着玻璃本身能阻挡60%的太阳辐射热。

---

## 2. 针对博莱利斯（Borealis）的热惯性模型
*Thermal Inertia Model for High-Latitude Heating*

在寒冷气候下，“捕获-存储-释放”是核心策略。模型重点修正了**热质量（Thermal Mass）** 的非稳态效应，解释了如何利用重质材料避免夏季过热并保存冬季热量。

### 2.1 热容阻尼方程 (Heat Capacity Damping)
我们在差分方程中大幅提高了室内热容项 $C_{in}$ 和墙体蓄热系数 $S = \sqrt{\rho c k}$。

室内空气温度变化的微分方程：

$$ C_{in} \frac{dT_{in}}{dt} = Q_{load} + \sum Q_{gains} $$

在 **Heavyweight（重质）** 场景中：
*   **$C_{in}$ 设定为 $2.0 \times 10^6 J/K$**（是轻质建筑的10倍）。这模拟了混凝土楼板、石材内墙等高蓄热材料。
*   物理效果表现为温度波动的**衰减（Decrement factor）**和**延迟（Time constant）**：
    $$ \tau_{thermal} = \frac{R_{total} \cdot C_{total}}{3600} \text{ (Hours)} $$
    重质建筑的 $\tau_{thermal}$ 极大，使得正午短暂的太阳得热被“锁定”在材料内部，并在夜间缓慢释放，从而平抑波动。

### 2.2 超级保温与过热预防 (Super-Insulation Paradox)
为了解释为什么高纬度也会过热，模型展示了当导热系数 $k$ 极小（$k=0.04$）时，室内产生的热量（$Q_{internal}$）和透过窗户的太阳辐射（$Q_{sol}$）无法通过墙体由于热阻过大而散发出去。

$$ Q_{loss} = \frac{A_{wall} \cdot (T_{in} - T_{out})}{R_{insulation}} \approx 0 \quad (\text{当 } R \to \infty) $$

在夏季，即使 $T_{out} = 15^\circ C$，如果 $Q_{internal} + Q_{sol}$ 很大且散热受阻，只有通过增加 $C_{in}$（热质量）来缓冲这些多余热量，公式体现为：

$$ \Delta T_{in} = \frac{(Q_{internal} + Q_{sol}) \cdot \Delta t}{C_{in}} $$

当 $C_{in}$ 很大时，$\Delta T_{in}$ 保持微小。这就是模型结果中重质结构能将夏季室温维持在 26°C 以下的数学原理，证明了不需要主动制冷即可度过暖季。

---

## 3. 论文引用建议

> "While the baseline model relies on steady-state assumptions for shading, the refined model integrates **transient thermodynamic behaviors**. For Sungrove, we implemented a **conditional advection algorithm** to simulate smart night ventilation (Equation 1) and modified the surface boundary condition to account for **albedo effects** (Equation 2). For Borealis, we adjusted the mass-matrix in the finite difference method to represent **high thermal inertia**, mathematically proving that increasing the characteristic time constant $\tau$ reduces the risk of summer overheating in super-insulated envelopes without active cooling."
