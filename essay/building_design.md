1. 总体布局与几何形态 (Orientation & Form)
朝向：建筑长轴呈东西向布置，最大化南北向表面积，最小化东西向表面积（东西向最难遮阳）。
形态：采用“回”字形或带有中央挑空中庭（Atrium）的紧凑布局。
南立面：作为主要的“集热与采光面”，但需严格控制。
北立面：布置不需要强光的辅助功能区（如储藏、卫生间），作为热缓冲区。
屋顶：设计为“蝴蝶形”或带有天窗，利于热空气排出和北向柔和采光。
2. 被动式设计五大原则的具体落实
① 开孔 (Aperture / Collector)

南向窗户：中等窗墙比 (WWR 30%-40%)。不仅是采光口，也是冬季供暖的入口。
北向窗户：较大窗墙比 (WWR 50-60%)，引入均匀的漫射光，无过热风险。
东西向：尽量少开窗，或者使用窄条长窗，防止低角度日晒导致下午过热。
顶部天窗：带有自动启闭功能的通风天窗。
② 控制 (Control / Shading)

南向遮阳：设置深远的水平遮阳板 (Overhangs) 或 光伏遮阳篷。计算好角度，夏季完全阻挡直射光，冬季允许阳光深入室内。
东西向遮阳：配置垂直遮阳百叶 (Vertical Fins) 或 动态穿孔金属板，阻导致眩光的低角度阳光。
植被遮阳：南向种植落叶乔木（夏季遮荫，冬季落叶透光）。
③ 吸收器 & ④ 热质量 (Absorber & Thermal Mass)

材料选择：中庭地面和底层墙体采用高热容材料（如深色抛光混凝土、夯土墙或石材）。
蓄热原理：
冬季：白天吸收穿过南向玻璃的阳光，储存热量。
夏季：由于有遮阳，热质不被直射，反而利用其热惰性吸收室内人员和设备产生的废热，保持室内凉爽，配合夜间通风散热。
⑤ 分配 (Distribution / Ventilation)

烟囱效应 (Stack Effect)：利用中央中庭作为排热通道。热空气上升通过顶部天窗排出，底部冷空气从侧窗进入，形成自然循环。
夜间通风 (Night Flushing)：夏季夜晚室外气温下降时，自动开启高处和低处通风口，利用穿堂风带走热质量白天吸收的热量，“清洗”建筑热蓄积。
3. 建筑围护结构与技术细节
高性能玻璃：选择 Low-E 玻璃，具备低 SHGC（太阳能热增益系数）以阻挡红外热辐射，但保持适宜的 VLT（可见光透过率）。
外墙保温：浅色外墙涂料（高反射率），减少结构吸热。

4. 解释
   Section 5: Optimization Results & Performance Evaluation
为了确定 Sungrove 大学新建学生中心的最佳被动式设计参数，我们建立了一个多目标优化框架。该框架综合考虑了热舒适度（Thermal Comfort）、能源消耗（Energy Efficiency）、初始建设成本（Construction Cost）以及视觉采光质量（Visual Quality）。

以下分析展示了通过模拟 720 种不同设计组合得出的关键结论。

5.1 Design Parameter Sensitivity: The "Valley" of Optimization
(对应图 1：Optimization Landscape / Contour Map)

为了探究窗墙比 (WWR) 与水平遮阳板深度 (Overhang Depth) 的耦合效应，我们绘制了总惩罚得分的等高线地形图（Figure 5.1）。

图形解读：
图表呈现出典型的 “山谷型”（Valley-shaped） 拓扑结构。颜色越蓝代表总惩罚分数越低（即综合性能越好），红色代表性能较差。
Global Optimum (全局最优解)：位于图中五角星标记处，对应 WWR 
≈
≈ 0.45-0.5 且 Overhang Depth 
≈
≈ 1.0-1.2m。
边界效应分析：
右下区域 (高 WWR, 无遮阳)：红色区域表明，当窗户过大且缺乏遮阳时，虽然采光充足，但夏季过热产生的热惩罚（Discomfort Penalty）急剧上升。
左上区域 (低 WWR, 深遮阳)：虽然隔热性能好，但丧失了冬季被动太阳能得热（Passive Solar Gain）和自然采光（Daylight），且深遮阳板带来了不必要的建筑成本，导致总分较差。
结论：
单纯追求“全封闭”或“全玻璃”都是不可取的。最优解证明了适度的开窗配合与其高度成比例的遮阳板（Shading Coefficient 
≈
≈ 0.5-0.6）是 Sungrove 气候条件下的最佳被动式策略。
5.2 Cost-Comfort Trade-off: Pareto Frontier Analysis
(对应图 2：Pareto Frontier)

由于预算限制和舒适度需求往往是冲突的，我们利用帕累托前沿分析（Pareto Frontier Analysis）来识别非支配解集（Non-dominated Solutions）。Figure 5.2 展示了不同设计方案在成本与舒适度维度上的分布。

图形解读：
X轴 (Cost)：初始建设成本（估算货币单位）。
Y轴 (Discomfort)：年平均不舒适度（度-小时数 > 26°C）。
气泡大小/颜色：代表年能源消耗总量 (MJ)。
关键发现：
Baseline (基准方案)：灰色叉号代表未优化的设计（如 WWR 0.5, 无遮阳），其位置表明虽然成本中等，但不舒适度极高，且能耗较大（气泡较大）。
边际收益递减 (Diminishing Returns)：帕累托前沿曲线呈凸型。从 Baseline 向 Optimal Solution (绿色五角星) 移动时，少量的成本投入（增加遮阳板）能带来舒适度的显著提升（Y轴大幅下降）。
然而，超过最优解后（向图表右下角移动），进一步增加遮阳板深度虽然能微弱提升舒适度，但建设成本呈线性上升，不再具备经济效益。
决策支持：
最优解（Optimal Solution）位于“肘部点”（Knee Point），代表了在每单位成本投入下获得最大舒适度回报的设计方案。
5.3 Holistic Comparison: Innovation vs. Tradition
(对应图 3：Radar Chart)

为了响应 COMAP 关于“下一代创新策略”的要求，我们将优化后的传统物理遮阳方案（Optimized Overhangs）与我们提出的创新方案——动态生物遮阳立面 (Green Facade) 进行了多维对比（Figure 5.3）。

维度定义：
Thermal Comfort & Energy Efficiency：物理性能指标。
Low Cost：经济性指标。
Ecology Score：生态价值（碳汇、生物多样性）。
Space/Aesthetics：空间与美学潜力。
对比分析：
Baseline (灰色)：典型的“玻璃盒子”建筑，除了建设成本较低外，在舒适度、能耗和生态性上得分均最低。
Optimized Traditional (蓝色)：通过物理遮阳优化，在热舒适度和能效上表现优异，但生态得分为中等。
Green Facade (绿色)：我们的创新方案。
利用落叶植物的季节性特性（夏季叶密遮阳，冬季落叶透光），在 Comfort 和 Energy 上达到了与精密物理遮阳相当甚至更好的效果（得益于植物蒸腾作用的微气候调节）。
在 Ecology Score 上具有压倒性优势，符合 Sungrove 2040 净零目标。
尽管维护成本可能略高，但其综合性能（雷达图覆盖面积最大）证明了它是最具前瞻性的选择。
在论文中的应用提示 (Tips for the Final Paper)
图注 (Captions)：

Fig 1: Contour map showing the total penalty score across the design space. The star indicates the global optimum, balancing thermal regulation and daylighting.
Fig 2: Pareto frontier of construction cost versus thermal discomfort. Bubble size represents annual energy consumption.
Fig 3: Multi-criteria radar chart comparing the baseline, optimized static shading, and the proposed green facade strategy.
关联 Problem Statement：

在解释图1时，提到 Sungrove 的低纬度高辐射特征，解释为什么不能完全不要遮阳。
在解释图3时，强调 "Student Union" 作为校园枢纽，不仅需要物理舒适，还需要生态展示功能（Green Facade 的优势）。
技术术语：

确保在正文中提到你使用了 Weighted Sum Method (加权和法) 来生成图1的等高线。
提到 Non-dominated Sorting (非支配排序) 来筛选图2中的帕累托最优解。